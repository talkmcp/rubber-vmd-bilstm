# Data Acquisition Guide

The raw market data used to produce the published results are licensed from third-party providers and **cannot be redistributed** by the authors under those terms. This document documents the data sources and access procedures so that researchers with their own provider access can independently assemble a CSV matching the documented schema.

For the exact column specification, see [`../data_schema/feature_dictionary.md`](../data_schema/feature_dictionary.md).

---

## Data sources by category

### 1. Thai rubber physical prices (8 columns)

`RSS3_FOBm1`, `RSS3_FOBm2`, `STR20_FOBm1`, `STR20_FOBm2`, `Latex_FOBm1`, `Latex_FOBm2`, `CupLump`, `USS`

- **Provider:** Thai Rubber Authority (TRA — การยางแห่งประเทศไทย, RAOT)
- **Coverage:** Daily Bangkok FOB quotations for ribbed smoked sheet, technical rubber, latex, cup lump, and unsmoked sheet rubber
- **Format:** Baht/kg
- **Access:** TRA daily market reports; some series are also redistributed via Bloomberg and Reuters terminals

### 2. International rubber futures (5 columns)

`RSS3_JPX_m1`, `RSS3_SHFE_m1`, `RSS3_SHFE_m2`, `RSS3_SGX_m1`, `TSR20_SGX_m2`

- **JPX / TOCOM (RSS3 m1):**
  - Provider: Japan Exchange Group (JPX) / Tokyo Commodity Exchange (TOCOM)
  - Format: JPY/kg
  - Access: JPX Data Cloud, Bloomberg, Refinitiv (Eikon/LSEG)

- **SHFE (RSS3 m1, m2):**
  - Provider: Shanghai Futures Exchange
  - Format: CNY/tonne
  - Access: SHFE official statistics; Bloomberg, Wind, CSMAR

- **SGX (RSS3 m1, TSR20 m2):**
  - Provider: Singapore Exchange (SICOM contracts)
  - Format: USD/kg (cents/kg in source — converted to USD/kg in our pipeline)
  - Access: SGX market data subscription; Bloomberg

### 3. Foreign exchange (3 columns)

`USD_THB`, `CNY_THB`, `USD_CNY`

- **Provider:** Bloomberg (preferred), Reuters/Refinitiv, or Bank of Thailand official daily rates
- **Format:** Local currency per foreign currency unit
- **Access:** Bloomberg Terminal `<USDTHB Curncy>` etc.; Bank of Thailand daily reference rate (free for some currencies)

### 4. Energy (4 columns)

`Brent`, `WTI`, `Brent_return`, `Brent_lag1`

- **Brent & WTI spot/near-month:**
  - Provider: U.S. Energy Information Administration (EIA), Reuters/Refinitiv
  - Format: USD/bbl
  - Access:
    - EIA: free public data via https://www.eia.gov/petroleum/
    - Reuters/Bloomberg: subscription
- **Brent_return** and **Brent_lag1** are computed inside the pipeline (log-return and t-1 lag of the level series respectively).

### 5. Macro / climate / policy (4 columns)

- **`China_PMI_Mfg`:**
  - Provider: CEIC Data, or directly from the National Bureau of Statistics of China
  - Format: index (mean ~50)
  - Frequency: monthly; forward-filled to daily inside the pipeline

- **`BDI` (Baltic Dry Index):**
  - Provider: Baltic Exchange via CEIC, Bloomberg, or Trading Economics
  - Format: index
  - Frequency: daily

- **`ENSO_ONI` (Oceanic Niño Index):**
  - Provider: NOAA Climate Prediction Center
  - Format: degrees Celsius anomaly
  - Frequency: monthly; forward-filled to daily
  - **Access:** free at https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php

- **`COVID_dummy`:**
  - Source: WHO COVID-19 timeline (publicly documented)
  - Binary indicator (0 / 1), defined as 1 during the WHO-declared Public Health Emergency of International Concern (30 Jan 2020 – 5 May 2023) for the affected portion of the Stage 3 sample
  - **Access:** free; this column is derived directly from public timelines

---

## Sample period

The Stage 3 sample used in the paper:

- **Start:** 7 May 2018
- **End:** 27 February 2026
- **Trading days:** 2,675 (raw); 175 after deduplication on the test partition

You may use any subset that matches the documented partition dates (see [`../data_schema/feature_dictionary.md`](../data_schema/feature_dictionary.md)), but to compare exactly with the published results, replicate the full Stage 3 window.

---

## Pre-processing performed inside the pipeline

The pipeline handles the following automatically — you do **not** need to do these in your CSV:

1. Forward-fill missing values from weekends/holidays (max 3 consecutive days)
2. Linear interpolation of remaining gaps (max 10 consecutive days)
3. First-differencing of the 21 non-stationary features (see ADF results in paper Appendix B)
4. MinMax scaling fitted on the training partition only (no leakage)
5. VMD decomposition with K = 6 modes, expanding-window protocol (see paper Section 2.2.5)

Your CSV should contain the **raw levels** (Baht/kg, USD/bbl, etc.) — the pipeline differences and normalises internally.

---

## License and redistribution

The data sources above are subject to their respective provider terms. Some (NOAA ENSO, WHO COVID, some EIA series) are public; most rubber-market and Bloomberg-derived series are subscription-licensed and may not be redistributed. The authors do not grant any license to the underlying data through this repository.

If you have provider access and would like to confirm whether a particular column construction matches the paper's exactly, please open an issue.
