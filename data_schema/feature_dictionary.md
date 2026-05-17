# Feature Dictionary — Stage 3 Input Data Schema

This document describes the **24 economic input features** used by the VMD-Augmented BiLSTM forecasting framework, plus the target variable. The pipeline expects a CSV file with these column names (case-sensitive) and the listed units. The Stage 3 sample covers **7 May 2018 – 27 February 2026** (n = 2,675 daily observations).

For data acquisition procedures, see [`../docs/DATA_ACQUISITION.md`](../docs/DATA_ACQUISITION.md).

---

## CSV column specification

The pipeline expects one row per trading day, sorted ascending by date. Missing values from weekends/holidays are forward-filled (max 3 consecutive days) then linearly interpolated (max 10 days) inside the pipeline.

| Column | Category | Unit | Source | Pre-processing |
|---|---|---|---|---|
| `Date` | identifier | YYYY-MM-DD | — | sort key; primary index |
| `RSS3_FOBm1` ★ | rubber spot (target) | Baht/kg | TRA | **first-difference** (Δp_t) |
| `RSS3_FOBm2` | rubber spot | Baht/kg | TRA | first-difference |
| `STR20_FOBm1` | rubber spot | Baht/kg | TRA | first-difference |
| `STR20_FOBm2` | rubber spot | Baht/kg | TRA | first-difference |
| `Latex_FOBm1` | rubber spot | Baht/kg | TRA | first-difference |
| `Latex_FOBm2` | rubber spot | Baht/kg | TRA | first-difference |
| `CupLump` | rubber spot | Baht/kg | TRA | first-difference |
| `USS` | rubber spot | Baht/kg | TRA | first-difference |
| `RSS3_JPX_m1` | rubber futures | JPY/kg | JPX/TOCOM | first-difference |
| `RSS3_SHFE_m1` | rubber futures | CNY/tonne | SHFE | first-difference |
| `RSS3_SHFE_m2` | rubber futures | CNY/tonne | SHFE | first-difference |
| `RSS3_SGX_m1` | rubber futures | USD/kg | SGX | first-difference |
| `TSR20_SGX_m2` | rubber futures | USD/kg | SGX | first-difference |
| `USD_THB` | FX | Baht/USD | Bloomberg | first-difference |
| `CNY_THB` | FX | Baht/CNY | Bloomberg | first-difference |
| `USD_CNY` | FX | CNY/USD | Bloomberg | first-difference |
| `Brent` | energy | USD/bbl | EIA / Reuters | first-difference |
| `WTI` | energy | USD/bbl | EIA / Reuters | first-difference |
| `Brent_return` | energy | proportion (Δlog) | derived | retained in returns |
| `Brent_lag1` | energy | USD/bbl | derived | first-difference of lag-1 Brent |
| `China_PMI_Mfg` | macro | index (~50) | CEIC | first-difference of monthly index, ffilled to daily |
| `BDI` | macro | index | CEIC | first-difference |
| `ENSO_ONI` | climate | °C anomaly | NOAA | retained in **levels** (climatological index) |
| `COVID_dummy` | policy | {0,1} | WHO | retained in **levels** (binary indicator) |

★ = target column. The first-differenced `Δp_t = p_t − p_{t−1}` of `RSS3_FOBm1` is the prediction target. The level series is also retained in the pipeline for price-level reconstruction (`p_hat_{t+1} = p_t + Δp_hat_{t+1}`).

**Total economic features: 24** (8 rubber spot + 5 rubber futures + 3 FX + 4 energy + 4 macro/climate). The 6 VMD IMF modes are constructed online from the differenced target and appended to give the 30-dimensional input vector per timestep.

---

## Data partitioning (chronological)

The pipeline splits the rows chronologically — **no shuffling** — into:

| Partition | Date range | Approx. rows | Purpose |
|---|---|---|---|
| Training | 2018-05-07 → 2023-08-27 | 2,140 (~80%) | Parameter estimation |
| Validation | 2023-08-28 → 2025-09-17 | 267 (~10%) | Early stopping, LR scheduling |
| Test (raw) | 2025-09-18 → 2026-02-27 | 237 (~9%) | Held-out evaluation |
| Test (deduplicated) | (subset of raw) | **175** | Primary metrics reported in paper |

After removing duplicated multi-contract entries on identical trading dates, the effective leakage-free test set used for all primary differenced-space evaluation is **n = 175** unique observations.

---

## Stationarity treatment

ADF unit-root tests on the training partition (lag order selected by AIC, max lags = 10, 5% significance):

- **21 of 24** features are non-stationary in levels (I(1)) — converted to first differences.
- **3 retained in levels**: `Brent_return` (already a stationary return), `China_PMI_Mfg` (mean-reverting index near 50), `ENSO_ONI` (bounded climate index), `COVID_dummy` (binary).

Full ADF results are reported in Appendix B of the paper.

---

## Synthetic sample CSV

A 100-row synthetic dummy file is provided at [`sample_synthetic_data.csv`](sample_synthetic_data.csv) for **smoke-testing the pipeline only**. It mirrors the schema but contains random values drawn from plausible distributions; it does **not** reproduce paper results.

To use real data: replace the synthetic CSV with a CSV matching this schema and re-run the training scripts.
