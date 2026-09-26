# Global Evaluation of AI and NWP Precipitation Forecasts During Atmospheric River Events

**Marina Vicens-Miquel, Taylor Mandelbaum, Amy McGovern, Aaron J. Hill, and Daniel Rothenberg**

This repository contains the code, data-processing workflows, and supporting resources associated with this study.

## Overview

Atmospheric rivers (ARs) produce many of the world's most extreme precipitation events and associated hydrometeorological hazards. Recent artificial intelligence weather prediction (AIWP) models have demonstrated skill comparable to or exceeding traditional numerical weather prediction (NWP) systems for several large-scale atmospheric variables. However, their ability to forecast precipitation associated with atmospheric rivers remains less well characterized globally.

This study provides a global evaluation of 24-hour precipitation forecasts from AIWP and NWP systems during atmospheric river events. Forecasts are evaluated from Day 1 through Day 10 over the globe and across three regions: North America, Europe, and Australia and New Zealand.

## Models

The evaluation includes the following forecasting systems:

- **GFS** — Global Forecast System
- **GEFS Mean** — Global Ensemble Forecast System ensemble mean
- **GraphCast** — AI-based global weather prediction model
- **AIFS** — Artificial Intelligence Forecasting System

Forecast precipitation is evaluated against **IMERG** (Integrated Multi-satellitE Retrievals for GPM) observations.

Atmospheric river events are identified using the **Extreme Weather Bench (EWB)** framework based on ERA5 data.

## Evaluation Period

The common evaluation period is:

**April 2021 – December 2024**

Forecasts initialized at 00, 06, 12, and 18 UTC are evaluated from **Day 1 through Day 10** using 24-hour accumulated precipitation.

## Evaluation Regions

Results are evaluated globally and over:

- North America
- Europe
- Australia and New Zealand

## Evaluation Metrics

Forecast performance is assessed using complementary measures of precipitation magnitude, spatial structure, and localization.

The evaluation includes:

- Precipitation bias
- Pearson correlation
- Fractions Skill Score (FSS)
- Critical Success Index (CSI)
- Precipitation probability distribution comparisons
- Spectral coherence

Threshold-based metrics are evaluated for rain/no-rain conditions and precipitation percentiles including the 75th, 90th, 95th, and 99th percentiles.

Uncertainty is quantified using event-based bootstrap confidence intervals.
