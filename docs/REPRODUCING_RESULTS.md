# Reproducing Paper Results

This document provides step-by-step instructions for reproducing the empirical results reported in:

> Pinitjitsamut, M. (2026). *Multi-Scale Forecasting of Natural Rubber Prices Using VMD-Augmented BiLSTM: A Hybrid Architecture Ablation Study.* **Forecasting** (MDPI).

---

## Hardware and runtime

The reference runtime environment used to produce the published results:

| Component | Specification |
|---|---|
| CPU | Intel Xeon / Apple M-series (any x86-64 or arm64 with AVX2) |
| RAM | 16 GB (32 GB recommended for multi-seed runs) |
| GPU | Optional. NVIDIA GPU with CUDA 11.8+ accelerates training ~3-5× |
| OS | Linux (Ubuntu 22.04), macOS 13+, Windows 11 (all tested) |

Single-seed training on CPU: **≈15–20 minutes** for the BiLSTM-only configuration on the Stage 3 training partition (n = 2,140). Full 5-seed protocol: **≈75–100 minutes** on CPU, ~15–25 minutes on a mid-range GPU.

---

## Step 1 — Environment setup

```bash
git clone https://github.com/talkmcp/rubber-vmd-bilstm.git
cd rubber-vmd-bilstm

# Python 3.11 strongly recommended
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -c "from vmdpy import VMD; print('vmdpy OK')"
```

---

## Step 2 — Obtain data

The raw market data are licensed third-party and cannot be redistributed (see [`DATA_ACQUISITION.md`](DATA_ACQUISITION.md)). You must:

1. Acquire access to: SGX, SHFE, JPX/TOCOM settlement series; Bloomberg-derived spot quotes for USD/THB, CNY/THB, USD/CNY, Brent, WTI; CEIC for China PMI and BDI; NOAA for ENSO ONI; TRA for Thai physical prices (RSS3, STR20, Latex, Cup Lump, USS).
2. Assemble a CSV file matching the schema in [`../data_schema/feature_dictionary.md`](../data_schema/feature_dictionary.md).
3. Save it at a path of your choice (referred to as `<DATA_CSV>` below).

For a smoke test, you can skip this step and use the included [`../data_schema/sample_synthetic_data.csv`](../data_schema/sample_synthetic_data.csv) — but it will not reproduce the paper numbers.

---

## Step 3 — Reproduce primary results (Table 11 / Section 3.4)

The multi-seed protocol trains the proposed VMD-Augmented BiLSTM under 5 independent random initialisations and reports mean ± SD on the deduplicated leakage-free test partition (n = 175).

```bash
python evaluation/multiseed_evaluation.py \
    --data <DATA_CSV> \
    --model bilstm_only \
    --seeds-file seeds.txt \
    --output-dir results/multiseed/bilstm_only/
```

**Expected output** (Table 11, paper Section 3.4):

```
DA%        = 82.5 ± 1.8
Pearson r  = 0.821 ± 0.016
StdR       = 1.091 ± 0.060
MAE_diff   = 0.339 ± 0.023 Baht/kg/day
RMSE_diff  = 0.456 ± 0.035 Baht/kg/day
```

Repeat for the Vanilla LSTM baseline:

```bash
python evaluation/multiseed_evaluation.py \
    --data <DATA_CSV> \
    --model vanilla_lstm \
    --seeds-file seeds.txt \
    --output-dir results/multiseed/vanilla_lstm/
```

**Expected output** (Table 13):

```
DA%        = 82.29 ± 0.00
Pearson r  = 0.398 ± 0.008
StdR       = 0.210 ± 0.007
```

The zero standard deviation of DA across 5 seeds is itself diagnostic of variance-collapse convergence (see paper Section 3.5).

---

## Step 4 — Reproduce ablation study (Table 10 / Section 3.3)

Single seed = 42, four architectural variants:

```bash
python evaluation/ablation_evaluation.py \
    --data <DATA_CSV> \
    --seed 42 \
    --output-dir results/ablation/
```

**Expected output** (Table 10, paper Section 3.3):

| Variant | DA% | r | StdR |
|---|---|---|---|
| Per-IMF BiLSTM+Transformer | 55.0 | 0.213 | 0.200 |
| VMD-as-features + BiLSTM only ★ | 83.4 | 0.838 | 1.029 |
| VMD-as-features + Transformer only | 72.8 | 0.635 | 0.680 |
| VMD-as-features + BiLSTM + Transformer | 67.5 | 0.659 | 0.697 |

---

## Step 5 — Reproduce multi-step forecasting (Table 14 / Section 3.6)

Direct strategy across 7 horizons, both architectures, single seed = 42:

```bash
python multistep/step9d_multistep_both.py \
    --data <DATA_CSV> \
    --horizons 1 2 3 5 10 20 30 \
    --seed 42 \
    --output-dir results/multistep/
```

**Expected output** (Table 14, paper Section 3.6) — BiLSTM-only rows:

| h | DA% | r | StdR |
|---|---|---|---|
| 1 | 82.1 | 0.827 | 0.985 |
| 2 | 82.7 | 0.835 | 0.827 |
| 3 | 74.5 | 0.738 | 0.682 |
| 5 | 71.4 | 0.486 | 0.475 |
| 10 | 68.8 | 0.239 | 0.265 |
| 20 | 56.3 | 0.130 | 0.126 |
| 30 | 52.3 | 0.285 | 0.007 |

---

## Step 6 — Supplementary analyses

ARIMA grid search (Appendix D):

```bash
python analysis/arima_grid_search.py --data <DATA_CSV> --output-dir results/arima/
```

VMD K-selection sensitivity (Appendix C):

```bash
python analysis/k_selection.py --data <DATA_CSV> --output-dir results/k_selection/
```

Look-back window L sensitivity (Appendix F):

```bash
python analysis/lookback_sensitivity.py --data <DATA_CSV> --output-dir results/lookback/
```

---

## Determinism notes

The pipeline enables PyTorch deterministic mode (`torch.use_deterministic_algorithms(True)`) and seeds NumPy, PyTorch CPU, and PyTorch CUDA (if available). Results should be reproducible to within floating-point round-off across runs on the **same hardware and PyTorch version**.

Small numerical drift (<0.005 on Pearson r, <0.01 on StdR) may occur across:

- Different PyTorch versions (within a major version)
- CPU vs GPU runs
- Different CUDA versions
- Different operating systems

These deviations are not material at the precision reported in the paper (3 decimal places).

---

## Troubleshooting

**`vmdpy` installation issue.** If pip fails on `vmdpy`, install from source:

```bash
pip install git+https://github.com/vrcarva/vmdpy.git
```

**Out-of-memory during training.** Reduce `--batch-size 16` (default is 32). The model fits comfortably in 4 GB RAM with batch=16.

**CUDA non-deterministic warnings.** Some CUDA operations are non-deterministic by design; with `torch.use_deterministic_algorithms(True)` these are converted to errors. If a non-deterministic op is required, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before running:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluation/multiseed_evaluation.py ...
```

---

## Citation

If you reproduce these results, please cite the paper (see [`../README.md`](../README.md)).
