"""
Atmospheric River Precipitation Evaluation
==========================================

Evaluates 24-hour precipitation forecasts from numerical weather prediction
(NWP) and AI weather prediction (AIWP) models against IMERG Final observations
during atmospheric river (AR) events.

The evaluation is implemented using the Extreme Weather Bench (EWB) framework
and supports GFS, GEFS ensemble mean, GraphCast, and AIFS forecasts across
multiple forecast lead times.

The pipeline performs temporal and spatial alignment of forecasts and
observations, constructs 24-hour precipitation accumulations, and evaluates
forecast performance using complementary metrics of precipitation magnitude,
spatial structure, and event localization. These include bias, Pearson
correlation, Fractions Skill Score (FSS), Critical Success Index (CSI),
precipitation distribution comparisons, and radial FFT-based spectral
coherence.

This script contains the primary evaluation workflow used to generate the
model-evaluation results presented in:

    "Global Evaluation of AI and NWP Precipitation Forecasts During
    Atmospheric River Events"

See the repository README for data access, preprocessing, and reproducibility
information.
"""

import os
import h5py
import s3fs
import yaml
import pickle
import scores
import inspect
import argparse 
import warnings
import icechunk
import dataclasses
import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from typing import Literal
import matplotlib.pyplot as plt
import extremeweatherbench as ewb
import cartopy.feature as cfeature
import scores.spatial as spatial_scores
from dask.diagnostics import ProgressBar
import extremeweatherbench.cases as cases
import extremeweatherbench.cases as ewb_cases
import extremeweatherbench.inputs as ewb_inputs
from extremeweatherbench.inputs import TargetBase, IncomingDataInput

# =============================================================================
# Environment configuration
# =============================================================================

DATASET_DIRECTORY = os.environ.get("DATASET_DIRECTORY")
if DATASET_DIRECTORY is None:
    raise EnvironmentError(
        "DATASET_DIRECTORY is not set. Set it to the root directory containing "
        "the precipitation datasets, atmospheric river event files, and "
        "evaluation outputs."
    )


# =============================================================================
# Evaluation constants
# =============================================================================

# Spatially uniform precipitation threshold (mm/24 h) used for the
# rain/no-rain FSS and CSI calculations. Percentile-based metrics instead use
# spatially varying IMERG climatological thresholds (p75, p90, p95, and p99).
RAIN_NO_RAIN_THRESHOLD_MM = 0.2

# =============================================================================
# Multiprocessing-safe metric storage
# =============================================================================

import multiprocessing as mp

# Shared storage for collecting metric outputs across parallel worker processes.
_storage_manager = None
global_spectrum_storage = None


def _init_global_storage():
    """Initialize shared storage for metric outputs from parallel workers."""
    global _storage_manager, global_spectrum_storage

    if _storage_manager is None:
        _storage_manager = mp.Manager()
        global_spectrum_storage = _storage_manager.dict({
            "SpectralCoherence": _storage_manager.list(),
            "SpectralCoherence_p90": _storage_manager.list(),
            "HistogramPDF": _storage_manager.list(),
        })


# =============================================================================
# Monthly data loading
# =============================================================================

def shift_month(month_str, offset):
    """
    Shift a YYYY_MM date string by a specified number of months.

    Examples
    --------
    shift_month("2021_01", -1) -> "2020_12"
    shift_month("2021_12",  1) -> "2022_01"
    """
    year, mon = month_str.split("_")
    dt = (
        pd.Timestamp(year=int(year), month=int(mon), day=1)
        + pd.DateOffset(months=offset)
    )
    return f"{dt.year}_{dt.month:02d}"


def try_open_zarr(zarr_path, label):
    """
    Attempt to open a Zarr store.

    Returns None if the store does not exist or cannot be opened. This allows
    optional adjacent-month data to be skipped without interrupting the
    evaluation.
    """
    if not os.path.exists(zarr_path):
        print(f"    ({label}: not found — {zarr_path})")
        return None

    try:
        ds = xr.open_zarr(zarr_path)
        print(f"    ({label}: loaded — {zarr_path})")
        return ds
    except Exception as e:
        print(f"    ({label}: failed to open — {e})")
        return None

 def imerg_path_for_month(path, month):
    """Return the path to the monthly IMERG precipitation Zarr store."""
    return (
        f"{path}/global_precip_evaluation/datasets/imerg_final/"
        f"tensors/{month}/imerg_hourly.zarr"
    )


def eval_dataset_path_for_month(path, dataset, month, lead_hours):
    """
    Return the path to a preprocessed monthly forecast Zarr store.

    GFS and GEFS ensemble-mean forecasts are loaded from preprocessed monthly
    tensors. GraphCast and AIFS are loaded directly from the MLWP Icechunk
    archive in `load_mlwp_24h_accumulated()`.
    """
    if dataset == "gfs":
        return (
            f"{path}/global_precip_evaluation/datasets/{dataset}/"
            f"precip_tensors_{lead_hours}h_accumulated/"
            f"{month[:-3]}/{month[-2:]}/gfs_hourly.zarr"
        )
    elif dataset == "gefs_mean":
        return (
            f"{path}/global_precip_evaluation/datasets/gefs/"
            f"precip_tensors_{lead_hours}h_accumulated_mean/"
            f"{month[:-3]}/{month[-2:]}/gefs_mean_hourly.zarr"
        )
    else:
        raise ValueError(
            f"Unknown dataset for on-disk tensor loading: {dataset}"
        )


def load_with_adjacent_months(path, month, path_fn, label):
    """
    Load data for the target month together with available adjacent months.

    Previous-month data provide the temporal context required to construct
    24-hour precipitation accumulations near the beginning of the target
    month. Next-month data are included because AR evaluation windows may
    extend across calendar-month boundaries.

    Adjacent-month data serve only as a temporal buffer; evaluation remains
    restricted to cases assigned to the target month, preventing duplicate
    evaluation across monthly runs. Data for the target month itself are
    required and a FileNotFoundError is raised if they are unavailable.
    """
    prev_month = shift_month(month, -1)
    next_month = shift_month(month, +1)

    print(
        f"  Loading {label} for {prev_month} (previous), "
        f"{month} (current), {next_month} (next)..."
    )

    datasets = []
    current_month_loaded = False

    for m in (prev_month, month, next_month):
        ds = try_open_zarr(path_fn(m), m)

        if ds is not None:
            datasets.append(ds)
            if m == month:
                current_month_loaded = True

    if not current_month_loaded:
        raise FileNotFoundError(
            f"{label} data for the target month {month} are unavailable. "
            "Adjacent-month buffer data alone are insufficient to evaluate "
            "the target month."
        )

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = xr.concat(datasets, dim="time").sortby("time")

        # Remove duplicate timestamps that may occur at monthly boundaries.
        _, unique_idx = np.unique(
            combined["time"].values,
            return_index=True,
        )
        combined = combined.isel(time=np.sort(unique_idx))

    return combined
# =============================================================================
# MLWP archive access
# =============================================================================
# GraphCast and AIFS forecasts are loaded directly from the MLWP Icechunk
# archive, where forecasts are organized by initialization time and lead time.
#
# Precipitation is stored as independent 6-hour accumulations rather than as
# cumulative precipitation since initialization. A 24-hour accumulation ending
# at lead time L is therefore constructed by summing the four most recent
# 6-hour precipitation increments:
#
#   accum_24h[L] = value[L] + value[L-6h] + value[L-12h] + value[L-18h]
#
# This produces 24-hour accumulations consistent with those used for the
# preprocessed GFS and GEFS datasets.

ICECHUNK_BUCKET = "brightband-public-mlwp-forecast-archive"
ICECHUNK_PREFIX_TMPL = "{model}.hres.icechunk"


def open_mlwp_archive(model: Literal["graphcast", "panguweather", "aifs-single"]) -> xr.Dataset:
    """Open the MLWP Icechunk archive for the specified model."""
    icechunk_prefix = ICECHUNK_PREFIX_TMPL.format(model=model)
    storage = icechunk.gcs_storage(bucket=ICECHUNK_BUCKET, prefix=icechunk_prefix)
    common_prefix = f"gs://{ICECHUNK_BUCKET}/{model}/hres/"
    gcs_credentials = icechunk.gcs_from_env_credentials()
    virtual_credentials = icechunk.containers_credentials({common_prefix: gcs_credentials})
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=virtual_credentials)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False, chunks="auto")
    return ds


def mlwp_precip_var_name(dataset):
    """Return the precipitation variable name used by the specified MLWP dataset."""
    return "total_precipitation_6hr" if dataset == "graphcast" else "total_precipitation"


def shift_month_timestamp(ts):
    """Return the first day of the month following the supplied timestamp."""
    return ts + pd.DateOffset(months=1)


def month_range_with_buffer(month):
    """
    Return the temporal range covering the previous, target, and next month.

    This three-month range provides the same temporal buffer used when loading
    the monthly GFS, GEFS, and IMERG datasets, ensuring consistent temporal
    coverage across data sources.
    """
    prev_month = shift_month(month, -1)
    next_next_month = shift_month(shift_month(month, +1), +1)

    prev_year, prev_mon = (int(x) for x in prev_month.split("_"))
    end_year, end_mon = (int(x) for x in next_next_month.split("_"))

    start = pd.Timestamp(year=prev_year, month=prev_mon, day=1)
    end_exclusive = pd.Timestamp(year=end_year, month=end_mon, day=1)
    return start, end_exclusive
def load_mlwp_24h_accumulated(dataset, month, lead_hours):
    """
    Construct 24-hour accumulated precipitation from the MLWP Icechunk archive.

    GraphCast and AIFS precipitation is stored as independent 6-hour
    accumulations indexed by initialization time and lead time. For each
    initialization time, the four most recent 6-hour accumulations
    (L, L-6 h, L-12 h, and L-18 h) are summed to produce the 24-hour
    accumulation ending at lead time L.

    The resulting data are organized by valid time at 6-hour intervals,
    consistent with the preprocessed GFS and GEFS datasets used in the
    evaluation. MLWP precipitation is converted from meters to millimeters
    to match the units of IMERG, GFS, GEFS, and the precipitation thresholds
    used by the evaluation metrics.
    """
    if lead_hours < 18:
        raise ValueError(
            f"lead_hours={lead_hours} is too small — need at least 18h of "
            f"lead time to sum 4x 6h chunks into a 24h accumulation."
        )

    var_name = mlwp_precip_var_name(dataset)
    print(f"  Loading {dataset} from MLWP icechunk archive (var={var_name})...")

    ds = open_mlwp_archive(dataset)
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in {dataset} archive. "
                        f"Available: {list(ds.data_vars)}")

    da = ds[var_name]

    start, end_exclusive = month_range_with_buffer(month)
    valid_times = pd.date_range(start=start, end=end_exclusive, freq="6h", inclusive="left")

    lead_td = pd.Timedelta(hours=lead_hours).to_numpy()
    init_times_needed = valid_times.values - lead_td

    available_init_times = da.init_time.values
    mask = np.isin(init_times_needed, available_init_times)
    n_missing = int((~mask).sum())
    if n_missing > 0:
        print(f"    ⚠️  {n_missing}/{len(init_times_needed)} required init_times not found "
              f"in {dataset} archive for this range — those valid_times will be skipped.")

    init_times_needed = init_times_needed[mask]
    valid_times_kept = valid_times.values[mask]

    if len(init_times_needed) == 0:
        raise FileNotFoundError(
            f"No usable init_times found for {dataset} covering {month} (with buffer)."
        )

    target_year, target_mon = (int(x) for x in month.split("_"))
    target_start = pd.Timestamp(year=target_year, month=target_mon, day=1)
    target_end_exclusive = shift_month_timestamp(target_start)
    kept_ts = pd.DatetimeIndex(valid_times_kept)
    n_in_target_month = int(((kept_ts >= target_start) & (kept_ts < target_end_exclusive)).sum())
    if n_in_target_month == 0:
        raise FileNotFoundError(
            f"❌ {dataset} data for the TARGET month {month} itself is missing "
            f"(prev/next-month buffer data alone is not sufficient to evaluate "
            f"{month}). Skipping this job rather than producing misleading "
            f"partial results."
        )

    lead_hours_needed = [lead_hours, lead_hours - 6, lead_hours - 12, lead_hours - 18]
    lead_tds_needed = [np.timedelta64(int(h * 3600), 's') for h in lead_hours_needed]

    print(f"    Selecting {len(init_times_needed)} init_times, summing lead_time "
          f"steps {lead_hours_needed}h to build 24h-accumulated precip...")

    da_sel = da.sel(init_time=init_times_needed, lead_time=lead_tds_needed)
    precip_24h = da_sel.sum(dim="lead_time", skipna=True).compute()

    # Convert precipitation from meters to millimeters for consistency with
    # IMERG, GFS, GEFS, and the precipitation thresholds used in the evaluation.
    precip_24h = precip_24h * 1000.0

    # Clip small negative values introduced by numerical precision to zero,
    # since accumulated precipitation is physically non-negative.
    precip_24h = precip_24h.clip(min=0.0)

    precip_24h = precip_24h.rename({"init_time": "time"})
    precip_24h = precip_24h.assign_coords(time=("time", valid_times_kept))
    precip_24h = precip_24h.sortby("time")

    result = precip_24h.to_dataset(name="precip")
    print(f"    ✓ Built {dataset} 24h-accumulated series: {len(result.time)} valid_time steps "
          f"(converted m->mm, negatives clipped to 0)")

    return result

# =============================================================================
# Radial spectral coherence
# =============================================================================

def get_radius_map(height, width):
    """Construct radial wavenumber bins for a two-dimensional FFT."""
    y = np.arange(-height // 2, height // 2)
    x = np.arange(-width // 2, width // 2)
    Y, X = np.meshgrid(y, x, indexing='ij')
    radius = np.round(np.sqrt(X**2 + Y**2)).astype(np.int32)
    max_radius = np.max(radius)
    return radius, max_radius


def compute_radial_profile_complex(power, radius_map, max_radius):
    """Compute the radial mean of a two-dimensional spectral field."""
    power_real = np.real(power)
    
    radial_profile = np.zeros(max_radius + 1, dtype=np.float32)
    for r in range(max_radius + 1):
        mask = (radius_map == r)
        if np.any(mask):
            radial_profile[r] = np.mean(power_real[mask])
    
    return radial_profile


def spectral_coherence_1d(pred, true, radius_map, max_radius):
    """
    Compute radially averaged spectral coherence between two spatial fields.

    The two-dimensional Fourier transforms of the forecast and reference
    fields are used to compute their cross- and auto-spectral densities.
    These spectra are radially averaged by wavenumber before calculating
    spectral coherence.

    Returns a one-dimensional coherence spectrum indexed by radial
    wavenumber.
    """
    # Replace non-finite values with zero before computing the Fourier transforms.
    valid = np.isfinite(pred) & np.isfinite(true)
    pred = np.where(valid, pred, 0.0).astype(np.float32)
    true = np.where(valid, true, 0.0).astype(np.float32)
    
    # Compute two-dimensional Fourier transforms.
    fft_pred = np.fft.fft2(pred)
    fft_true = np.fft.fft2(true)
    
    # Compute cross- and auto-spectral densities.
    S_xy = fft_pred * np.conj(fft_true)
    S_xx = fft_pred * np.conj(fft_pred)
    S_yy = fft_true * np.conj(fft_true)
    
    # Radially average the spectral fields by wavenumber.
    Sxy_r = compute_radial_profile_complex(S_xy, radius_map, max_radius)
    Sxx_r = compute_radial_profile_complex(S_xx, radius_map, max_radius)
    Syy_r = compute_radial_profile_complex(S_yy, radius_map, max_radius)
    
    # Compute spectral coherence from the radially averaged spectra.
    coherence = np.abs(Sxy_r)**2 / (Sxx_r * Syy_r + 1e-8)
    coherence = np.where(np.isnan(coherence), 0.0, coherence)
    
    return coherence


def mean_spectral_coherence_1d(pred_batch, true_batch):
    """
    Compute the mean radial spectral coherence across a batch of spatial fields.

    Individual coherence spectra are calculated for each forecast-reference
    pair and then averaged across the batch.
    """
    pred_batch = np.asarray(pred_batch, dtype=np.float32)
    true_batch = np.asarray(true_batch, dtype=np.float32)
    
    r_shape = pred_batch.ndim
    
    if r_shape == 2:
        pred_batch = pred_batch[np.newaxis, :, :]
        true_batch = true_batch[np.newaxis, :, :]
    elif r_shape == 3 and pred_batch.shape[-1] == 1:
        pred_batch = np.squeeze(pred_batch, axis=-1)[np.newaxis, :, :]
        true_batch = np.squeeze(true_batch, axis=-1)[np.newaxis, :, :]
    elif r_shape == 4 and pred_batch.shape[-1] == 1:
        pred_batch = np.squeeze(pred_batch, axis=-1)
        true_batch = np.squeeze(true_batch, axis=-1)
    
    N, H, W = pred_batch.shape
    radius_map, max_radius = get_radius_map(H, W)
    
    spectra = []
    for i in range(N):
        spec = spectral_coherence_1d(pred_batch[i], true_batch[i], radius_map, max_radius)
        spectra.append(spec)
    
    return np.mean(spectra, axis=0)


def plot_spectral_coherence_batch(spectra_list, output_path, batch_id, dataset, lead_time, coherence_type="overall"):
    """Plot and save spectral coherence curves for a batch of evaluation cases."""
    os.makedirs(output_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_list)))
    
    for idx, spectrum in enumerate(spectra_list):
        spectrum_np = np.asarray(spectrum)
        x_vals = np.arange(len(spectrum_np))[::-1]
        spectrum_plot = spectrum_np[::-1]
        
        case_num = (batch_id - 1) * 50 + idx + 1
        ax.plot(x_vals, spectrum_plot, label=f'Case {case_num}', 
               color=colors[idx], linewidth=1.2, alpha=0.75)
    
    ax.set_xlabel('Wavenumber (# oscillations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spectral Coherence', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 250)
    ax.set_yscale('log')
    
    title = f'Spectral Coherence vs Wavenumber - {dataset.upper()} ({coherence_type.upper()})\nBatch {batch_id} (Cases {(batch_id-1)*50 + 1}-{(batch_id-1)*50 + len(spectra_list)}) - Lead {lead_time}h'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, 
             frameon=True, fancybox=True, shadow=True, ncol=1)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_path, 
                            f'spectral_coherence_batch{batch_id}_{coherence_type}_{dataset}_{lead_time}h.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


# =============================================================================
# Histogram-based precipitation PDF metric
# =============================================================================

def compute_relative_frequency_pdf(values, bin_edges):
    """Compute a normalized histogram representing relative frequency by bin."""
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    
    if len(values) == 0:
        return np.full(len(bin_edges) - 1, np.nan, dtype=np.float32)
    
    hist, _ = np.histogram(values, bins=bin_edges)
    
    total = np.sum(hist)
    if total == 0:
        return np.full(len(hist), np.nan, dtype=np.float32)
    
    relative_freq = hist.astype(np.float32) / total
    return relative_freq


def get_first_valid_frame(forecast_np, target_np):
    """
    Return the first forecast-target pair for which the target contains
    nonzero precipitation.
    """
    if forecast_np.ndim == 4:
        forecast_np = forecast_np[..., 0]
    if target_np.ndim == 4:
        target_np = target_np[..., 0]
    
    for t in range(target_np.shape[0]):
        if np.any(target_np[t] > 0):
            return forecast_np[t], target_np[t]
    
    return None, None


def plot_pdf_batch(pdf_list, output_path, batch_id, dataset, lead_time, bin_edges):
    """Plot and save forecast and target precipitation PDFs for a batch of cases."""
    os.makedirs(output_path, exist_ok=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(pdf_list)))
    
    ax = axes[0]
    for idx, pdf_dict in enumerate(pdf_list):
        fc_pdf = np.asarray(pdf_dict['forecast_pdf'], dtype=np.float32)
        case_num = (batch_id - 1) * 50 + idx + 1
        
        if not np.all(np.isnan(fc_pdf)):
            ax.plot(bin_centers, fc_pdf, label=f'Case {case_num}', 
                   color=colors[idx], linewidth=1.2, alpha=0.7)
    
    ax.set_ylabel('Relative Frequency (PDF)', fontsize=11, fontweight='bold')
    ax.set_title(f'Forecast PDFs - {dataset.upper()} - Batch {batch_id} - Lead {lead_time}h', 
                fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1)
    
    ax = axes[1]
    for idx, pdf_dict in enumerate(pdf_list):
        ob_pdf = np.asarray(pdf_dict['target_pdf'], dtype=np.float32)
        case_num = (batch_id - 1) * 50 + idx + 1
        
        if not np.all(np.isnan(ob_pdf)):
            ax.plot(bin_centers, ob_pdf, label=f'Case {case_num}', 
                   color=colors[idx], linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel('Rainfall (mm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (PDF)', fontsize=11, fontweight='bold')
    ax.set_title(f'Target PDFs - {dataset.upper()} - Batch {batch_id} - Lead {lead_time}h', 
                fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_path, 
                             f'pdf_batch{batch_id}_{dataset}_{lead_time}h.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


class HistogramBasedPDF(ewb.metrics.BaseMetric):
    """
    Histogram-based precipitation distribution metric.

    Forecast and target precipitation values are represented as relative-
    frequency histograms using 40 bins spanning 0–20 mm. The metric is the
    root-mean-square difference between the forecast and target relative
    frequencies across histogram bins.
    """
    
    def __init__(self, preserve_dims="lead_time", name="HistogramPDF", **kwargs):
        super().__init__(name=name, preserve_dims=preserve_dims, **kwargs)
        self.bin_edges = np.linspace(0.0, 20.0, 41, dtype=np.float32)
        self.call_count = 0
        print(f"✓ Initialized {name} metric (bins: 0-20mm, 40 bins)")
    
    def _compute_metric(self, forecast, target, **kwargs):
        global global_spectrum_storage
        
        self.call_count += 1
        
        try:
            fc_np = forecast.compute().values if hasattr(forecast.data, 'compute') else forecast.values
            ob_np = target.compute().values if hasattr(target.data, 'compute') else target.values
            
            fc_2d, ob_2d = get_first_valid_frame(fc_np, ob_np)
            
            if fc_2d is None or ob_2d is None:
                nan_pdf = np.full(len(self.bin_edges) - 1, np.nan, dtype=np.float32)
                global_spectrum_storage[self.name].append({
                    'forecast_pdf': nan_pdf.copy(),
                    'target_pdf': nan_pdf.copy(),
                    'pdf_distance': np.nan
                })
                
                lead_coord = (forecast.coords["lead_time"].values
                              if "lead_time" in forecast.coords else [0])
                return xr.DataArray(
                    [np.nan],
                    dims=["lead_time"],
                    coords={"lead_time": lead_coord},
                )
            
            fc_flat = fc_2d.flatten()
            ob_flat = ob_2d.flatten()
            
            fc_pdf = compute_relative_frequency_pdf(fc_flat, self.bin_edges)
            ob_pdf = compute_relative_frequency_pdf(ob_flat, self.bin_edges)
            
            pdf_distance = np.sqrt(np.nanmean((fc_pdf - ob_pdf)**2))
            
            global_spectrum_storage[self.name].append({
                'forecast_pdf': fc_pdf.copy(),
                'target_pdf': ob_pdf.copy(),
                'pdf_distance': float(pdf_distance)
            })
            
            lead_coord = (forecast.coords["lead_time"].values
                          if "lead_time" in forecast.coords else [0])
            
            return xr.DataArray(
                [pdf_distance],
                dims=["lead_time"],
                coords={"lead_time": lead_coord},
            )
        
        except Exception as e:
            print(f"❌ ERROR in {self.name}._compute_metric (case {self.call_count}): {e}")
            import traceback
            traceback.print_exc()
            lead_coord = (forecast.coords["lead_time"].values
                          if "lead_time" in forecast.coords else [0])
            return xr.DataArray(
                [np.nan],
                dims=["lead_time"],
                coords={"lead_time": lead_coord},
            )


# =============================================================================
# Spectral coherence metric for Extreme Weather Bench
# =============================================================================

class SpectralCoherence(ewb.metrics.BaseMetric):
    """
    Radial spectral coherence metric based on two-dimensional Fourier transforms.

    When a threshold raster is provided, coherence is computed using only
    locations where the target precipitation meets or exceeds the spatially
    varying threshold. Otherwise, coherence is computed over the full field.
    The returned metric is the mean of the radial coherence spectrum.
    """
    
    def __init__(self, threshold_raster=None, preserve_dims="lead_time", 
                 name="SpectralCoherence", output_path=None, **kwargs):
        super().__init__(name=name, preserve_dims=preserve_dims, **kwargs)
        self.preserve_dims = preserve_dims
        self.output_path = output_path
        self.threshold_raster = threshold_raster
        self.call_count = 0
        threshold_type = 'p90' if threshold_raster is not None else 'overall'
        print(f"✓ Initialized {name} metric (threshold_type={threshold_type})")
    
    def _compute_metric(self, forecast, target, **kwargs):
        global global_spectrum_storage
        
        self.call_count += 1
        
        try:
            fc_np = forecast.compute().values if hasattr(forecast.data, 'compute') else forecast.values
            ob_np = target.compute().values if hasattr(target.data, 'compute') else target.values
            
            nlat = forecast.sizes.get("latitude", fc_np.shape[-2])
            nlon = forecast.sizes.get("longitude", fc_np.shape[-1])
            
            fc_np = fc_np.reshape(-1, nlat, nlon)
            ob_np = ob_np.reshape(-1, nlat, nlon)
            
            if self.threshold_raster is not None:
                threshold_np = (
                    self.threshold_raster
                    .sel(latitude=forecast.latitude, longitude=forecast.longitude,
                         method="nearest")
                    .transpose("latitude", "longitude")
                    .values
                )
                mask = ob_np >= threshold_np[np.newaxis, :, :]
                fc_np_masked = np.where(mask, fc_np, 0.0)
                ob_np_masked = np.where(mask, ob_np, 0.0)
            else:
                fc_np_masked = fc_np
                ob_np_masked = ob_np
            
            spectrum = mean_spectral_coherence_1d(fc_np_masked, ob_np_masked)
            mean_coherence = np.nanmean(spectrum)
            
            global_spectrum_storage[self.name].append({
                'spectrum': spectrum.copy(),
                'mean_coherence': float(mean_coherence)
            })
            
            lead_coord = (forecast.coords["lead_time"].values
                          if "lead_time" in forecast.coords else [0])
            
            return xr.DataArray(
                [mean_coherence],
                dims=["lead_time"],
                coords={"lead_time": lead_coord},
            )
        
        except Exception as e:
            print(f"❌ ERROR in {self.name}._compute_metric (case {self.call_count}): {e}")
            import traceback
            traceback.print_exc()
            lead_coord = (forecast.coords["lead_time"].values
                          if "lead_time" in forecast.coords else [0])
            return xr.DataArray(
                [np.nan],
                dims=["lead_time"],
                coords={"lead_time": lead_coord},
            )

def restructure_forecast_for_ewb(ds: xr.Dataset, var_name: str, lead_hours: int) -> xr.Dataset:
    """
    Restructure forecast data into the dimensions and coordinates required by
    Extreme Weather Bench.

    Valid times are converted to initialization times using the specified
    forecast lead time, and a singleton lead-time dimension is added while
    preserving the latitude and longitude coordinates.
    """
    valid_times = ds.time.values
    lead_td = np.timedelta64(lead_hours, "h")
    init_times = valid_times - lead_td
    lead_time = np.array([pd.Timedelta(hours=lead_hours).to_timedelta64()])
    data_expanded = ds[var_name].data[:, np.newaxis, ...]

    return xr.Dataset(
        {
            var_name: (
                ["init_time", "lead_time", "latitude", "longitude"],
                data_expanded,
            )
        },
        coords={
            "init_time": init_times,
            "valid_time": ("init_time", valid_times),
            "lead_time": lead_time,
            "latitude": ds.latitude.values,
            "longitude": ds.longitude.values,
        },
    )


def restructure_target_for_ewb(ds: xr.Dataset, var_name: str, lead_hours: int) -> xr.Dataset:
    """
    Restructure target data into the dimensions and coordinates required by
    Extreme Weather Bench.

    Initialization times are derived from the target valid times and specified
    forecast lead time so that observations are temporally aligned with the
    corresponding forecasts during evaluation.
    """
    valid_times = ds.time.values
    lead_td = np.timedelta64(lead_hours, "h")
    init_times = valid_times - lead_td
    lead_time = np.array([pd.Timedelta(hours=lead_hours).to_timedelta64()])
    data_expanded = ds[var_name].data[:, np.newaxis, ...]

    return xr.Dataset(
        {
            var_name: (
                ["init_time", "lead_time", "latitude", "longitude"],
                data_expanded,
            )
        },
        coords={
            "init_time": init_times,
            "valid_time": ("init_time", valid_times),
            "lead_time": lead_time,
            "latitude": ds.latitude.values,
            "longitude": ds.longitude.values,
        },
    )


class MaskedMeanError(ewb.MeanError):
    """
    Mean error computed where target precipitation exceeds a spatially varying
    threshold.

    The threshold raster is applied to the target field, and the mean error is
    evaluated only at locations meeting or exceeding that threshold.
    """

    def __init__(self, threshold_raster: xr.DataArray,
                 preserve_dims="lead_time", name="MeanError", **kwargs):
        super().__init__(name=name, preserve_dims=preserve_dims, **kwargs)
        self.threshold_raster = threshold_raster

    def _compute_metric(self, forecast, target, **kwargs):
        mask     = target >= self.threshold_raster
        forecast = forecast.where(mask)
        target   = target.where(mask)
        return super()._compute_metric(forecast, target, **kwargs)


class MaskedMeanAbsoluteError(ewb.MeanAbsoluteError):
    """
    Mean absolute error computed where target precipitation exceeds a spatially
    varying threshold.

    The threshold raster is applied to the target field, and the mean absolute
    error is evaluated only at locations meeting or exceeding that threshold.
    """

    def __init__(self, threshold_raster: xr.DataArray,
                 preserve_dims="lead_time", name="MeanAbsoluteError", **kwargs):
        super().__init__(name=name, preserve_dims=preserve_dims, **kwargs)
        self.threshold_raster = threshold_raster

    def _compute_metric(self, forecast, target, **kwargs):
        mask     = target >= self.threshold_raster
        forecast = forecast.where(mask)
        target   = target.where(mask)
        return super()._compute_metric(forecast, target, **kwargs)


class PearsonCorrelation(ewb.metrics.BaseMetric):
    """Pearson correlation between forecast and target precipitation fields."""

    def __init__(self, preserve_dims="lead_time", name="PearsonCorrelation", **kwargs):
        super().__init__(name=name, preserve_dims=preserve_dims, **kwargs)
        self.preserve_dims = preserve_dims

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ):
        return scores.continuous.correlation.pearsonr(
            fcst          = forecast,
            obs           = target,
            preserve_dims = self.preserve_dims,
        )


class RasterThresholdFSS(ewb.metrics.BaseMetric):
    """
    Fractions Skill Score (FSS) using a spatially varying precipitation threshold.

    Forecast and target fields are converted to binary exceedance fields using
    the supplied threshold raster. Neighborhood event fractions are then
    calculated over the specified window, and FSS is computed from the
    differences between forecast and observed neighborhood fractions.

    The neighborhood window is reduced when necessary for domains smaller than
    the requested window size.
    """

    def __init__(self, threshold_raster, window_size=(15,15),
                 preserve_dims="lead_time", name="FSS", **kwargs):
        super().__init__(name, preserve_dims=preserve_dims, **kwargs)
        self.threshold_raster = threshold_raster
        self.window_size      = window_size
        self.preserve_dims    = preserve_dims

    @staticmethod
    def _fss_numpy_2d(fc_2d: np.ndarray, ob_2d: np.ndarray,
                      threshold: np.ndarray, window: tuple) -> float:
        """Compute FSS for two spatial fields using a raster-based threshold."""
        fc_bin  = (fc_2d >= threshold).astype(float)
        ob_bin  = (ob_2d >= threshold).astype(float)

        def sliding_sum(arr, win):
            """Compute moving-window sums using a two-dimensional cumulative sum."""
            r, c   = win
            cs     = np.cumsum(np.cumsum(arr, axis=0), axis=1)
            cs_pad = np.pad(cs, ((1, 0), (1, 0)), mode='constant')
            return (cs_pad[r:, c:] - cs_pad[:-r, c:] -
                    cs_pad[r:, :-c] + cs_pad[:-r, :-c])

        fc_frac = sliding_sum(fc_bin, window) / (window[0] * window[1])
        ob_frac = sliding_sum(ob_bin, window) / (window[0] * window[1])
        num     = np.nansum((fc_frac - ob_frac) ** 2)
        denom   = np.nansum(fc_frac ** 2) + np.nansum(ob_frac ** 2)
        return float(1.0 - num / denom) if denom > 0 else np.nan

    def _compute_metric(self, forecast, target, **kwargs):
        import warnings

        fc_np = forecast.compute().values if hasattr(forecast.data, 'compute') else forecast.values
        ob_np = target.compute().values   if hasattr(target.data,   'compute') else target.values

        nlat = forecast.sizes["latitude"]
        nlon = forecast.sizes["longitude"]
        fc_np = fc_np.reshape(-1, nlat, nlon)
        ob_np = ob_np.reshape(-1, nlat, nlon)

        win_lat = max(1, min(self.window_size[0], nlat - 1))
        win_lon = max(1, min(self.window_size[1], nlon - 1))
        window  = (win_lat, win_lon)

        if window != self.window_size:
            warnings.warn(
                f"{self.name}: window clipped {self.window_size}→{window} "
                f"(domain {nlat}×{nlon})")

        threshold_np = (
            self.threshold_raster
            .sel(latitude=forecast.latitude, longitude=forecast.longitude,
                 method="nearest")
            .transpose("latitude", "longitude")
            .values
        )

        fss_values = [
            self._fss_numpy_2d(fc_np[t], ob_np[t], threshold_np, window)
            for t in range(fc_np.shape[0])
        ]

        lead_coord = (forecast.coords["lead_time"].values
                      if "lead_time" in forecast.coords else [0])
        return xr.DataArray(
            [np.nanmean(fss_values)],
            dims=["lead_time"],
            coords={"lead_time": lead_coord},
        )


@dataclasses.dataclass
class XarrayTarget(TargetBase):
    """
    In-memory xarray target compatible with the Extreme Weather Bench
    evaluation interface.
    """

    ds: xr.Dataset = None
    source: str = "memory"
    name: str = "in-memory target"

    def __post_init__(self):
        if self.ds is None:
            raise ValueError("'ds' is required for XarrayTarget.")

    def _open_data_from_source(self) -> xr.Dataset:
        return self.ds

    def subset_data_to_case(self, data, case_metadata, **kwargs):
        return ewb_inputs.ForecastBase.subset_data_to_case(
            self, data, case_metadata, **kwargs
        )


@dataclasses.dataclass
class XarrayForecastNoLeadTime(ewb.XarrayForecast):
    """In-memory xarray forecast wrapper used by the EWB evaluation workflow."""
    pass


class RasterThresholdCSI(ewb.metrics.CriticalSuccessIndex):
    """
    Critical Success Index (CSI) using a spatially varying precipitation threshold.

    The same threshold raster is applied to the forecast and target fields to
    define precipitation exceedances.
    """

    def __init__(self, threshold_raster: xr.DataArray,
                 preserve_dims="lead_time", **kwargs):
        super().__init__(
            preserve_dims    = preserve_dims,
            forecast_threshold = threshold_raster,
            target_threshold   = threshold_raster,
            **kwargs
        )


def percentile_aligned(percentiles, percentile_key, eval_dataset_lat_res, 
                       lat_factor, lon_factor, eval_dataset_final):
    """
    Align an IMERG precipitation-percentile raster with the evaluation grid.

    Percentile thresholds are provided in mm/24 h on the native 0.1-degree
    IMERG grid, with dimensions ordered as (latitude, longitude). The
    percentile field is converted to 0–360-degree longitude coordinates,
    coarsened to the evaluation resolution using block maxima, and reindexed
    to the evaluation grid using nearest-neighbor matching.

    No temporal scaling is applied because the percentile thresholds are
    computed directly from 24-hour accumulated IMERG precipitation.
    """
    lat_imerg = np.arange(-89.95, 90.0,  0.1).astype(np.float32)   # Native IMERG latitude grid.
    lon_imerg = np.arange(-179.95, 180.0, 0.1).astype(np.float32)  # Native IMERG longitude grid.

    p_imerg = xr.DataArray(
        percentiles[percentile_key],
        coords={"latitude": lat_imerg, "longitude": lon_imerg},
        dims=["latitude", "longitude"]   # Percentile arrays are stored as (latitude, longitude).
    )

    p_aligned = (
        p_imerg
        .assign_coords(longitude=((p_imerg.longitude + 360) % 360))
        .sortby("longitude")
        .sortby("latitude")
        .coarsen(latitude=lat_factor, longitude=lon_factor, boundary="trim")
        .max()
        .reindex(
            latitude  = eval_dataset_final.latitude,
            longitude = eval_dataset_final.longitude,
            method    = "nearest",
            tolerance = eval_dataset_lat_res,
        )
    )

    return p_aligned.astype(np.float32)


def compute_n_pixels_per_case(pkl_path, mask_key='core_mask_union'):
    """
    Return the number of AR-mask grid cells associated with each evaluation case.

    Pixel counts are computed from the AR masks stored in the original pickle
    file because the YAML case definitions contain only the bounding regions
    used by Extreme Weather Bench. The returned dictionary maps each evaluation
    case identifier to the number of grid cells contained in its AR mask.
    """
    with open(pkl_path, 'rb') as f:
        events_dict = pickle.load(f)

    return {
        eval_id: int(np.asarray(event[mask_key].values).sum())
        for eval_id, event in events_dict.items()
    }


def converting_to_yaml(pkl_path):
    """
    Convert AR evaluation-window metadata from pickle format to YAML case
    definitions compatible with Extreme Weather Bench.

    Each case retains its evaluation identifier, temporal window, event type,
    and geographic bounding region derived from the AR object bounds.
    """
    with open(pkl_path, "rb") as f:
        eval_events = pickle.load(f)

    case_list = []

    for case_id, (eval_id, ev) in enumerate(sorted(eval_events.items()), start=1):
        bounds = ev["largest_object_bounds"]

        start_dt = pd.Timestamp(ev["window_start"]).to_pydatetime()
        end_dt   = pd.Timestamp(ev["window_end"]).to_pydatetime()

        case = {
            "case_id_number": case_id,
            "title":          eval_id,                          
            "start_date":     start_dt,   
            "end_date":       end_dt,     
            "event_type":     "atmospheric_river",
            "location": {
                "type": "bounded_region",
                "parameters": {
                    "latitude_min":  float(bounds["latitude_min"]),
                    "latitude_max":  float(bounds["latitude_max"]),
                    "longitude_min": float(bounds["longitude_min"]),
                    "longitude_max": float(bounds["longitude_max"]),
                }
            }
        }
        case_list.append(case)

    print(f"Total cases: {len(case_list)}")
    print(f"\nFirst case:\n{case_list[0]}")

    out_path = pkl_path.split('.')[0] + '.yml'

    with open(out_path, "w") as f:
        yaml.dump(case_list, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def computing_24h_accum_precip(imerg_aligned, eval_dataset_final):
    """
    Compute 24-hour accumulated precipitation at 6-hour intervals.

    Both datasets include data from the adjacent months to provide sufficient
    temporal context at calendar-month boundaries. Previous-month data support
    complete 24-hour accumulation windows near the beginning of the target
    month, while next-month data allow AR evaluation windows that cross a
    calendar-month boundary to be retained.
    """
    
    print("\n" + "="*70)
    print("IMERG Processing - Frequency Detection")
    print("="*70)
    
    imerg_time_diff = imerg_aligned.time.diff(dim="time")
    imerg_time_diff_hours = imerg_time_diff / np.timedelta64(1, 'h')
    imerg_time_diffs_computed = imerg_time_diff_hours.compute()
    imerg_diffs_values = imerg_time_diffs_computed.values
    
    imerg_expected_freq_hours = np.nanmedian(imerg_diffs_values)
    print(f"IMERG temporal frequency: {imerg_expected_freq_hours:.2f}h")
    
    imerg_gap_threshold = imerg_expected_freq_hours * 2.0
    imerg_gaps = imerg_diffs_values > imerg_gap_threshold
    
    imerg_24h_raw = (
        imerg_aligned
        .rolling(time=24, min_periods=1)
        .sum()
    )
    
    imerg_24h_raw = imerg_24h_raw.resample(time="6h", label="right", closed="right").first()
    
    imerg_count = (
        imerg_aligned
        .rolling(time=24, min_periods=1)
        .count()
    )
    imerg_count = imerg_count.resample(time="6h", label="right", closed="right").first()
    
    expected_count_imerg = int(24 / imerg_expected_freq_hours)

    imerg_count_computed = imerg_count.compute()
    imerg_counts_flat = imerg_count_computed.values.flatten()
    
    zero_count = (imerg_counts_flat == 0).sum()
    if zero_count > 0:
        print(f"⚠️  {zero_count} windows with 0 timesteps")
    
    imerg_complete_mask = (imerg_count_computed > 0) & (imerg_count_computed >= expected_count_imerg)
    
    n_complete_imerg = imerg_complete_mask.sum().item()
    n_total_imerg = len(imerg_24h_raw.time)
    
    imerg_24h = imerg_24h_raw.compute().where(imerg_complete_mask, drop=True)
    
    print("\n" + "="*70)
    print("eval_dataset Processing - Frequency Detection")
    print("="*70)
    
    eval_time_diff = eval_dataset_final["precip"].time.diff(dim="time")
    eval_time_diff_hours = eval_time_diff / np.timedelta64(1, 'h')
    eval_time_diffs_computed = eval_time_diff_hours.compute()
    eval_diffs_values = eval_time_diffs_computed.values
    
    eval_expected_freq_hours = np.nanmedian(eval_diffs_values)
    print(f"Detected eval_dataset frequency: {eval_expected_freq_hours:.2f}h")
    
    eval_gap_threshold = eval_expected_freq_hours * 2.0
    eval_gaps = eval_diffs_values > eval_gap_threshold
    
    eval_24h_raw = (
        eval_dataset_final["precip"]
        .rolling(time=4, min_periods=1)
        .sum()
    )
    
    eval_24h_raw = eval_24h_raw.resample(time="6h", label="right", closed="right").first()
    
    eval_count = (
        eval_dataset_final["precip"]
        .rolling(time=4, min_periods=1)
        .count()
    )
    eval_count = eval_count.resample(time="6h", label="right", closed="right").first()
    
    expected_count_eval = int(24 / eval_expected_freq_hours)
    
    eval_count_computed = eval_count.compute()
    eval_counts_flat = eval_count_computed.values.flatten()
    
    zero_count_eval = (eval_counts_flat == 0).sum()
    if zero_count_eval > 0:
        print(f"⚠️  {zero_count_eval} windows with 0 timesteps")
    
    eval_complete_mask = (eval_count_computed > 0) & (eval_count_computed >= expected_count_eval)
    
    n_complete_eval = eval_complete_mask.sum().item()
    n_total_eval = len(eval_24h_raw.time)
    
    eval_dataset_24h = eval_24h_raw.compute().where(eval_complete_mask, drop=True)
    
    print("\n" + "="*70)
    print("Time Alignment")
    print("="*70)
    
    common_times = imerg_24h.time.values[np.isin(
        imerg_24h.time.values, 
        eval_dataset_24h.time.values
    )]
    
    n_common = len(common_times)
    print(f"Common 6-hourly windows: {n_common}")
    print(f"(Range now spans previous/current/next month buffer, so this count "
          f"will be larger than a single month's ~124 windows — that's expected.)")
    
    if n_common == 0:
        raise ValueError(
            "❌ No common windows between IMERG and eval_dataset. "
            "Cannot proceed with evaluation."
        )
    
    imerg_24h = imerg_24h.sel(time=common_times)
    eval_dataset_24h = eval_dataset_24h.sel(time=common_times)
    
    if not np.array_equal(imerg_24h.time.values, eval_dataset_24h.time.values):
        raise ValueError("❌ CRITICAL: Time coordinates not identical after alignment.")
    
    print(f"✓ Both datasets aligned to {n_common} common 6-hourly windows")
    print(f"  First window: {imerg_24h.time.values[0]}")
    print(f"  Last window:  {imerg_24h.time.values[-1]}")
    print("="*70 + "\n")

    # Reduce spatial completeness masks to one value per timestep before
    # reporting the number of complete temporal windows.
    print("\n" + "="*70)
    print("DATA COMPLETENESS DIAGNOSTICS")
    print("="*70)

    non_time_dims_imerg = [d for d in imerg_complete_mask.dims if d != "time"]
    non_time_dims_eval  = [d for d in eval_complete_mask.dims if d != "time"]

    imerg_complete_1d = (imerg_complete_mask.all(dim=non_time_dims_imerg)
                          if non_time_dims_imerg else imerg_complete_mask)
    eval_complete_1d  = (eval_complete_mask.all(dim=non_time_dims_eval)
                          if non_time_dims_eval else eval_complete_mask)

    n_complete_imerg_1d = int(imerg_complete_1d.sum())
    n_complete_eval_1d  = int(eval_complete_1d.sum())

    print(f"IMERG complete windows: {n_complete_imerg_1d} / {n_total_imerg} "
          f"({100*n_complete_imerg_1d/n_total_imerg:.1f}%)")
    print(f"Eval dataset complete windows: {n_complete_eval_1d} / {n_total_eval} "
          f"({100*n_complete_eval_1d/n_total_eval:.1f}%)")

    imerg_times = set(imerg_24h.time.values)
    eval_times = set(eval_dataset_24h.time.values)

    only_in_imerg = imerg_times - eval_times
    only_in_eval  = eval_times - imerg_times

    print(f"\nTimes with valid IMERG but missing/incomplete eval dataset: {len(only_in_imerg)}")
    if only_in_imerg:
        print(f"  First few: {sorted(only_in_imerg)[:5]}")

    print(f"\nTimes with valid eval dataset but missing/incomplete IMERG: {len(only_in_eval)}")
    if only_in_eval:
        print(f"  First few: {sorted(only_in_eval)[:5]}")
    print("="*70 + "\n")
    
    return imerg_24h, eval_dataset_24h


def saving_histogram_pdf_results(pdf_path, dataset, month, lead_hours):
    """
    Save case-level histogram-based precipitation distribution results.

    Forecast and target relative-frequency histograms, together with their
    corresponding distribution-distance metric, are stored in pickle format.
    Summary statistics and bin configuration are additionally written to a
    YAML metadata file.
    """
    global global_spectrum_storage
    
    print("\n" + "="*70)
    print("SAVING HISTOGRAM PDF RESULTS")
    print("="*70)

    pdf_list = global_spectrum_storage['HistogramPDF']
    if pdf_list:
        forecast_pdfs = [p['forecast_pdf'] for p in pdf_list]
        target_pdfs = [p['target_pdf'] for p in pdf_list]
        pdf_distances = np.array([p['pdf_distance'] for p in pdf_list])
        
        pdf_pkl = f'{pdf_path}/histogram_pdf_{dataset}_{month}_{lead_hours}h.pkl'
        pkl_data = {
            'forecast_pdfs': forecast_pdfs,
            'target_pdfs': target_pdfs,
            'pdf_distances': pdf_distances,
            'dataset': dataset,
            'month': month,
            'lead_hours': lead_hours,
            'n_cases': len(pdf_list),
            'bin_edges': np.linspace(0.0, 20.0, 41, dtype=np.float32),
            'timestamp': str(pd.Timestamp.now())
        }
        
        with open(pdf_pkl, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"✓ Saved {len(pdf_list)} PDFs to PKL: {os.path.basename(pdf_pkl)}")
        
        pdf_yml = pdf_pkl.replace('.pkl', '.yml')
        yml_metadata = {
            'file_type': 'histogram_pdf_results',
            'dataset': dataset,
            'month': month,
            'lead_hours': lead_hours,
            'n_cases': len(pdf_list),
            'timestamp': str(pd.Timestamp.now()),
            'statistics': {
                'pdf_distances': {
                    'min': float(np.nanmin(pdf_distances)),
                    'max': float(np.nanmax(pdf_distances)),
                    'mean': float(np.nanmean(pdf_distances)),
                    'std': float(np.nanstd(pdf_distances)),
                    'median': float(np.nanmedian(pdf_distances))
                }
            },
            'bin_config': {
                'min_value_mm': 0.0,
                'max_value_mm': 20.0,
                'n_bins': 40,
                'bin_width_mm': 0.5
            },
            'format': 'pickle',
            'note': 'Each forecast_pdf and target_pdf has shape (40,). PDF distance = L2 distance between PDFs.'
        }
        
        with open(pdf_yml, 'w') as f:
            yaml.dump(yml_metadata, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved metadata to YAML: {os.path.basename(pdf_yml)}")
        
    else:
        print("⚠️  No histogram PDF results found!")
    
    print("="*70)


def saving_spectral_coherence_results(spectral_coherence, spectral_coherence_p90, 
                                      spectral_coh_path, dataset, month, lead_hours):
    """
    Save case-level radial spectral coherence results.

    Overall and p90-thresholded coherence spectra are stored separately in
    pickle files. Corresponding YAML files contain summary statistics and
    metadata describing each collection of spectra.
    """
    global global_spectrum_storage
    
    print("\n" + "="*70)
    print("SAVING SPECTRAL COHERENCE RESULTS")
    print("="*70)

    spectra_list = global_spectrum_storage['SpectralCoherence']
    if spectra_list:
        mean_coherences = np.array([s['mean_coherence'] for s in spectra_list])
        
        spectra_pkl = f'{spectral_coh_path}/spectral_coherence_overall_{dataset}_{month}_{lead_hours}h.pkl'
        pkl_data = {
            'spectra': [s['spectrum'].copy() for s in spectra_list],
            'mean_coherences': mean_coherences,
            'dataset': dataset,
            'month': month,
            'lead_hours': lead_hours,
            'n_cases': len(spectra_list),
            'timestamp': str(pd.Timestamp.now())
        }
        
        with open(spectra_pkl, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"✓ Saved {len(spectra_list)} overall spectra to PKL: {os.path.basename(spectra_pkl)}")
        
        spectra_yml = spectra_pkl.replace('.pkl', '.yml')
        yml_metadata = {
            'file_type': 'spectral_coherence_results',
            'coherence_type': 'overall',
            'dataset': dataset,
            'month': month,
            'lead_hours_hours': lead_hours,
            'n_cases': len(spectra_list),
            'timestamp': str(pd.Timestamp.now()),
            'statistics': {
                'mean_coherence': {
                    'min': float(np.min(mean_coherences)),
                    'max': float(np.max(mean_coherences)),
                    'mean': float(np.mean(mean_coherences)),
                    'std': float(np.std(mean_coherences)),
                    'median': float(np.median(mean_coherences))
                },
                'spectrum_lengths': {
                    'min': min(len(s['spectrum']) for s in spectra_list),
                    'max': max(len(s['spectrum']) for s in spectra_list),
                    'mean': np.mean([len(s['spectrum']) for s in spectra_list])
                }
            },
            'format': 'pickle',
            'note': 'Each spectrum can have different length (variable max_radius). Load with: pickle.load(open(pkl_file, "rb"))'
        }
        
        with open(spectra_yml, 'w') as f:
            yaml.dump(yml_metadata, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved metadata to YAML: {os.path.basename(spectra_yml)}")
        
    else:
        print("⚠️  No overall spectral coherence results found!")
    
    spectra_list_p90 = global_spectrum_storage['SpectralCoherence_p90']
    if spectra_list_p90:
        mean_coherences_p90 = np.array([s['mean_coherence'] for s in spectra_list_p90])
        
        spectra_pkl_p90 = f'{spectral_coh_path}/spectral_coherence_p90_{dataset}_{month}_{lead_hours}h.pkl'
        pkl_data_p90 = {
            'spectra': [s['spectrum'].copy() for s in spectra_list_p90],
            'mean_coherences': mean_coherences_p90,
            'dataset': dataset,
            'month': month,
            'lead_hours': lead_hours,
            'n_cases': len(spectra_list_p90),
            'timestamp': str(pd.Timestamp.now())
        }
        
        with open(spectra_pkl_p90, 'wb') as f:
            pickle.dump(pkl_data_p90, f)
        print(f"✓ Saved {len(spectra_list_p90)} p90 spectra to PKL: {os.path.basename(spectra_pkl_p90)}")
        
        spectra_yml_p90 = spectra_pkl_p90.replace('.pkl', '.yml')
        yml_metadata_p90 = {
            'file_type': 'spectral_coherence_results',
            'coherence_type': 'p90_threshold',
            'dataset': dataset,
            'month': month,
            'lead_hours': lead_hours,
            'n_cases': len(spectra_list_p90),
            'timestamp': str(pd.Timestamp.now()),
            'statistics': {
                'mean_coherence': {
                    'min': float(np.min(mean_coherences_p90)),
                    'max': float(np.max(mean_coherences_p90)),
                    'mean': float(np.mean(mean_coherences_p90)),
                    'std': float(np.std(mean_coherences_p90)),
                    'median': float(np.median(mean_coherences_p90))
                },
                'spectrum_lengths': {
                    'min': min(len(s['spectrum']) for s in spectra_list_p90),
                    'max': max(len(s['spectrum']) for s in spectra_list_p90),
                    'mean': np.mean([len(s['spectrum']) for s in spectra_list_p90])
                }
            },
            'format': 'pickle',
            'note': 'P90 threshold applied. Each spectrum can have different length. Load with: pickle.load(open(pkl_file, "rb"))'
        }
        
        with open(spectra_yml_p90, 'w') as f:
            yaml.dump(yml_metadata_p90, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved metadata to YAML: {os.path.basename(spectra_yml_p90)}")
        
    else:
        print("⚠️  No p90 spectral coherence results found!")
    
    print("="*70)


def get_base_path():
    """Return the root directory containing the evaluation datasets and outputs."""
    return DATASET_DIRECTORY


def get_ar_eval_events_dir(path, eval_type):
    """Return the directory containing AR evaluation-window definitions."""
    return f"{path}/global_precip_evaluation/ar_eval_events/{eval_type}"


def get_results_base_dir(dataset, month, lead_hours, eval_type):
    """
    Return the output directory for a dataset, month, lead time, and
    evaluation region.

    Results are organized by evaluation region, dataset, year, month, and
    forecast lead time.
    """
    path = get_base_path()
    return f'{path}/global_precip_evaluation/precip_model_results_for_ar/{eval_type}/{dataset}/{month[:-3]}/{month}/{lead_hours}h'


def get_output_csv_path(dataset, month, lead_hours, eval_type):
    """
    Return the CSV output path for an individual evaluation run.

    Each combination of dataset, month, lead time, and evaluation region is
    assigned a unique output path. The same path is used when saving results
    and when checking whether an evaluation has already been completed.
    """
    out_path = get_results_base_dir(dataset, month, lead_hours, eval_type)
    return f'{out_path}/ar_{dataset}_eval_{month}_{lead_hours}h.csv'


def model_evaluation(month, lead_hours, dataset, eval_type):
    """
    Run the atmospheric-river precipitation evaluation using Extreme Weather
    Bench.

    The evaluation region determines the AR case definitions used for the run
    and the corresponding output directory. Forecasts and IMERG observations
    are temporally and spatially aligned before computing precipitation
    magnitude, spatial-skill, distribution, and spectral-coherence metrics.
    """
    global _storage_manager, global_spectrum_storage
    
    _init_global_storage()

    path = get_base_path()

    percentiles = np.load(f'{path}/global_precip_evaluation/datasets/imerg_final/percentile_rasters/percentiles.npz')
    
    # Load the target month together with adjacent-month data to provide
    # temporal context for 24-hour accumulations and AR windows that cross
    # calendar-month boundaries. Evaluation remains restricted to cases
    # assigned to the target month.
    print("\n" + "="*70)
    print("LOADING DATA (current month + adjacent months buffer)")
    print("="*70)

    imerg = load_with_adjacent_months(
        path, month,
        path_fn=lambda m: imerg_path_for_month(path, m),
        label="IMERG",
    )

    if dataset in ('gfs', 'gefs_mean'):
        # Load preprocessed monthly GFS or GEFS precipitation tensors using
        # the same adjacent-month buffer applied to IMERG.
        eval_dataset = load_with_adjacent_months(
            path, month,
            path_fn=lambda m: eval_dataset_path_for_month(path, dataset, m, lead_hours),
            label=dataset,
        )
    elif dataset in ('graphcast', 'aifs-single'):
        # Load GraphCast or AIFS directly from the MLWP archive over the same
        # three-month temporal range and construct 24-hour accumulations from
        # the underlying 6-hour precipitation fields.
        eval_dataset = load_mlwp_24h_accumulated(dataset, month, lead_hours)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print("="*70 + "\n")

    # Align longitude conventions and sort spatial coordinates.
    imerg_final = (
        imerg
        .assign_coords(longitude=((imerg.longitude + 360) % 360))
        .sortby("longitude")
        .sortby("latitude")
    )
    eval_dataset_final = eval_dataset.sortby("latitude")

    # Coarsen IMERG to the forecast-grid resolution using block maxima and
    # align the resulting field with the forecast coordinates.
    imerg_lat_res     = float(imerg_final.latitude[1]  - imerg_final.latitude[0])
    imerg_lon_res     = float(imerg_final.longitude[1] - imerg_final.longitude[0])
    eval_dataset_lat_res = float(eval_dataset_final.latitude[1]  - eval_dataset_final.latitude[0])
    eval_dataset_lon_res = float(eval_dataset_final.longitude[1] - eval_dataset_final.longitude[0])

    lat_factor = int(round(eval_dataset_lat_res / imerg_lat_res))
    lon_factor = int(round(eval_dataset_lon_res / imerg_lon_res))

    imerg_coarse = (
        imerg_final["precip"]
        .coarsen(latitude=lat_factor, longitude=lon_factor, boundary="trim")
        .max()
    )

    imerg_aligned = imerg_coarse.reindex(
        latitude  = eval_dataset_final.latitude,
        longitude = eval_dataset_final.longitude,
        method    = "nearest",
        tolerance = eval_dataset_lat_res,
    )

    # Construct and temporally align 24-hour precipitation accumulations.
    imerg_24h, eval_dataset_24h = computing_24h_accum_precip(imerg_aligned, eval_dataset_final)

    imerg_ds_24h = imerg_24h.to_dataset(name="precip")
    eval_dataset_ds_24h = eval_dataset_24h.to_dataset(name="precip")

    imerg_ewb = restructure_target_for_ewb(imerg_ds_24h, "precip", lead_hours)
    eval_dataset_ewb = restructure_forecast_for_ewb(eval_dataset_ds_24h, "precip", lead_hours)

    # Construct Extreme Weather Bench target and forecast objects.
    custom_target = XarrayTarget(
        ds               = imerg_ewb,
        name             = "IMERG Final",
        variables        = ["accumulated_precipitation"],  
        variable_mapping = {"precip": "accumulated_precipitation"},
    )

    custom_forecast = XarrayForecastNoLeadTime(
        ds               = eval_dataset_ewb,
        name             = dataset,
        variables        = ["accumulated_precipitation"],   
        variable_mapping = {"precip": "accumulated_precipitation"},
    )

    # Load AR evaluation cases assigned to the target month. Each evaluation
    # window is assigned to a single monthly case file according to its start
    # time, preventing duplicate evaluation across monthly runs.
    ar_month = month.replace('_', '')

    ar_eval_events_dir = get_ar_eval_events_dir(path, eval_type)
    pkl_path = f"{ar_eval_events_dir}/eval_events_complete_{ar_month}_final.pkl"
    yml_path = f"{ar_eval_events_dir}/eval_events_complete_{ar_month}_final.yml"

    if not os.path.exists(yml_path):
        converting_to_yaml(pkl_path)

    with open(yml_path, "r") as f:
        loaded = yaml.safe_load(f)

    cases_yaml = ewb.load_individual_cases(loaded)

    # Retrieve the number of AR-mask grid cells associated with each case.
    n_pixels_lookup = compute_n_pixels_per_case(pkl_path)

    # Align spatially varying IMERG percentile thresholds with the forecast grid.
    p75_aligned = percentile_aligned(percentiles, "p75", eval_dataset_lat_res, lat_factor, lon_factor, eval_dataset_final)
    p90_aligned = percentile_aligned(percentiles, "p90", eval_dataset_lat_res, lat_factor, lon_factor, eval_dataset_final)
    p95_aligned = percentile_aligned(percentiles, "p95", eval_dataset_lat_res, lat_factor, lon_factor, eval_dataset_final)
    p99_aligned = percentile_aligned(percentiles, "p99", eval_dataset_lat_res, lat_factor, lon_factor, eval_dataset_final)

    # Construct a spatially uniform rain/no-rain threshold on the same grid as
    # the percentile thresholds.
    rain_norain_aligned = xr.full_like(p75_aligned, RAIN_NO_RAIN_THRESHOLD_MM)

    # Initialize evaluation metrics.
    correlation = PearsonCorrelation(preserve_dims="lead_time", name="PearsonCorrelation")

    spectral_coh_path = f'{get_results_base_dir(dataset, month, lead_hours, eval_type)}/spectral_coherence_{month}_{lead_hours}h'
    os.makedirs(spectral_coh_path, exist_ok=True)  
    spectral_coherence = SpectralCoherence(threshold_raster=None, preserve_dims="lead_time", 
                                          name="SpectralCoherence", output_path=spectral_coh_path)  
    spectral_coherence_p90 = SpectralCoherence(threshold_raster=p90_aligned, preserve_dims="lead_time",
                                              name="SpectralCoherence_p90", output_path=spectral_coh_path)

    pdf_path = f'{get_results_base_dir(dataset, month, lead_hours, eval_type)}/histogram_pdf_{month}_{lead_hours}h'
    os.makedirs(pdf_path, exist_ok=True)
    histogram_pdf = HistogramBasedPDF(preserve_dims="lead_time", name="HistogramPDF")

    bias_p75 = MaskedMeanError(threshold_raster=p75_aligned, preserve_dims="lead_time", name="MeanError_p75")
    bias_p90 = MaskedMeanError(threshold_raster=p90_aligned, preserve_dims="lead_time", name="MeanError_p90")
    bias_p95 = MaskedMeanError(threshold_raster=p95_aligned, preserve_dims="lead_time", name="MeanError_p95")
    bias_p99 = MaskedMeanError(threshold_raster=p99_aligned, preserve_dims="lead_time", name="MeanError_p99")

    mae_p75  = MaskedMeanAbsoluteError(threshold_raster=p75_aligned, preserve_dims="lead_time", name="MeanAbsoluteError_p75")
    mae_p90  = MaskedMeanAbsoluteError(threshold_raster=p90_aligned, preserve_dims="lead_time", name="MeanAbsoluteError_p90")
    mae_p95  = MaskedMeanAbsoluteError(threshold_raster=p95_aligned, preserve_dims="lead_time", name="MeanAbsoluteError_p95")
    mae_p99  = MaskedMeanAbsoluteError(threshold_raster=p99_aligned, preserve_dims="lead_time", name="MeanAbsoluteError_p99")

    fss_p75 = RasterThresholdFSS(
        threshold_raster = p75_aligned,
        window_size      = (15, 15),
        preserve_dims    = "lead_time",
        name             = "FSS_p75",
    )
    fss_p90 = RasterThresholdFSS(
        threshold_raster = p90_aligned,
        window_size      = (15, 15),
        preserve_dims    = "lead_time",
        name             = "FSS_p90",
    )
    fss_p95 = RasterThresholdFSS(
        threshold_raster = p95_aligned,
        window_size      = (15, 15),
        preserve_dims    = "lead_time",
        name             = "FSS_p95",
    )
    fss_p99 = RasterThresholdFSS(
        threshold_raster = p99_aligned,
        window_size      = (15, 15),
        preserve_dims    = "lead_time",
        name             = "FSS_p99",
    )

    csi_p75 = RasterThresholdCSI(threshold_raster=p75_aligned,
                                   preserve_dims="lead_time",         
                                   name="CSI_p75")

    csi_p90 = RasterThresholdCSI(threshold_raster=p90_aligned,
                                   preserve_dims="lead_time",
                                   name="CSI_p90")

    csi_p95 = RasterThresholdCSI(threshold_raster=p95_aligned,
                                   preserve_dims="lead_time",
                                   name="CSI_p95")

    csi_p99 = RasterThresholdCSI(threshold_raster=p99_aligned, 
                                   preserve_dims="lead_time",
                                   name="CSI_p99")

    # Compute overall mean error without a precipitation threshold.
    bias_overall = ewb.MeanError(preserve_dims="lead_time", name="MeanError_overall")

    # Compute FSS and CSI using the spatially uniform rain/no-rain threshold.
    fss_rain_norain = RasterThresholdFSS(
        threshold_raster = rain_norain_aligned,
        window_size      = (15, 15),
        preserve_dims    = "lead_time",
        name             = "FSS_rain_norain",
    )

    csi_rain_norain = RasterThresholdCSI(threshold_raster=rain_norain_aligned,
                                          preserve_dims="lead_time",
                                          name="CSI_rain_norain")

    # Assemble the Extreme Weather Bench evaluation configuration.
    ar_evaluation_objects = [
        ewb.EvaluationObject(
            event_type="atmospheric_river",
            metric_list=[
                bias_overall,
                correlation,
                spectral_coherence,
                histogram_pdf,
                bias_p75, bias_p90, bias_p95, bias_p99,
                csi_p75,  csi_p90,  csi_p95,  csi_p99,
                fss_p75,  fss_p90,  fss_p95,  fss_p99,
                csi_rain_norain,
                fss_rain_norain,
                spectral_coherence_p90
            ],
            target=custom_target,
            forecast=custom_forecast,
        ),
    ]

    # Run the Extreme Weather Bench evaluation.
    ar_ewb = ewb.ExtremeWeatherBench(
        case_metadata      = cases_yaml,
        evaluation_objects = ar_evaluation_objects,
    )

    case_lookup = {
        c.case_id_number: {
            "case_title": c.title,
            "eval_start": pd.Timestamp(c.start_date),
            "eval_end": pd.Timestamp(c.end_date),
            "event_type_case": c.event_type,
        }
        for c in cases_yaml
    }

    print("\n" + "="*70)
    print("RUNNING EWB EVALUATION (20 parallel jobs)")
    print("="*70)
    
    outputs = ar_ewb.run(parallel_config={"backend": "loky", "n_jobs": 20})

    print(f"\n✓ Spectral coherence collection complete:")
    print(f"  - Overall: {len(global_spectrum_storage['SpectralCoherence'])} spectra")
    print(f"  - P90: {len(global_spectrum_storage['SpectralCoherence_p90'])} spectra")
    
    print(f"\n✓ Histogram PDF collection complete:")
    print(f"  - PDFs: {len(global_spectrum_storage['HistogramPDF'])} cases")

    outputs["case_title"] = outputs["case_id_number"].map(lambda x: case_lookup[x]["case_title"])
    outputs["eval_start"] = outputs["case_id_number"].map(lambda x: case_lookup[x]["eval_start"])
    outputs["eval_end"] = outputs["case_id_number"].map(lambda x: case_lookup[x]["eval_end"])

    outputs["forecast_init_time"] = outputs["eval_start"] - pd.to_timedelta(lead_hours, unit="h")

    outputs["lead_time_hours"] = lead_hours

    # Add the number of AR-mask grid cells associated with each evaluation case.
    outputs["n_pixels"] = outputs["case_title"].map(n_pixels_lookup)

    # Reshape the EWB output from one row per metric to one row per case, with
    # individual metrics represented as columns.
    id_cols = [
        "case_id_number",
        "case_title",
        "eval_start",
        "eval_end",
        "forecast_init_time",
        "lead_time",
        "lead_time_hours",
        "n_pixels",
        "forecast_source",
        "target_source",
    ]

    outputs = outputs.pivot_table(
        index=id_cols, columns="metric", values="value", aggfunc="first"
    ).reset_index()
    outputs.columns.name = None

    # Place case metadata first and metric columns afterward in a stable order.
    metric_cols = sorted(c for c in outputs.columns if c not in id_cols)
    outputs = outputs[id_cols + metric_cols]

    saving_spectral_coherence_results(spectral_coherence, spectral_coherence_p90, 
                                      spectral_coh_path, dataset, month, lead_hours)

    saving_histogram_pdf_results(pdf_path, dataset, month, lead_hours)

    # Save the final case-level evaluation results.
    out_csv_path = get_output_csv_path(dataset, month, lead_hours, eval_type)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    outputs.to_csv(out_csv_path, index=False)

    print(f"\n✓ Saved evaluation results to: {out_csv_path}")


def main():
    """Parse command-line arguments and run the AR precipitation evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate precipitation forecasts during atmospheric river events "
            "using the Extreme Weather Bench framework."
        )
    )
    parser.add_argument("dataset",  type=str, help="Enter the dataset that we want to evaluate (gfs, gefs_mean, graphcast, aifs-single)")
    parser.add_argument("month",  type=str, help="Enter the month that we want to evaluate (ex. 2024_02)")
    parser.add_argument("lead_hours",  type=int, help="Enter the lead hours that we want to evaluate (ex. 240)")
    parser.add_argument("eval_type", type=str,
                        choices=["global", "europe", "north_america", "australia_new_zealand"],
                        help=("Enter the evaluation type: 'global' for the standard "
                              "whole-globe evaluation, or one of the three case "
                              "studies ('europe', 'north_america', "
                              "'australia_new_zealand'). Case studies evaluate only "
                              "AR events that made landfall inside that region's "
                              "box, anchored at landfall time rather than event "
                              "start, clipped to the region's spatial extent."))

    args  = parser.parse_args()
    dataset = args.dataset
    month  = args.month
    lead_hours = args.lead_hours
    eval_type = args.eval_type
    print(f'Evaluating {dataset} - {month} - lead time: {lead_hours}h - eval_type: {eval_type}')

    # Skip the evaluation when the output for this exact combination of
    # dataset, month, lead time, and evaluation region already exists.
    expected_output_csv = get_output_csv_path(dataset, month, lead_hours, eval_type)
    if os.path.exists(expected_output_csv):
        print(f"\n✓ Output already exists — job already completed successfully: {expected_output_csv}")
        print("  Skipping re-run.")
        return

    model_evaluation(month, lead_hours, dataset, eval_type)


if __name__ == "__main__":
    main()
    