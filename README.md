# Multi-Scale Forecasting of Natural Rubber Prices Using VMD-Augmented BiLSTM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Companion code repository for the article:

> **Pinitjitsamut, M.** (2026). *Multi-Scale Forecasting of Natural Rubber Prices Using VMD-Augmented BiLSTM: A Hybrid Architecture Ablation Study.* **Forecasting** (MDPI).

## Overview

This repository contains the source code, training configurations, and evaluation pipeline for a deep-learning forecasting framework that predicts daily first-differenced **RSS3 FOB natural rubber prices** by combining **Variational Mode Decomposition (VMD)** with a **bidirectional LSTM (BiLSTM)** encoder. The contribution is methodological:

1. **VMD-as-features.** Rather than forecasting each intrinsic mode function independently and aggregating, all six IMF series are appended directly to the 24-feature economic input matrix, preserving multi-scale information within a single forward pass.
2. **Leakage-free expanding-window VMD.** Decomposition for validation and test observations uses only information up to the forecast origin τ; no observation from τ+1 or later enters either the decomposition or the input look-back window.
3. **Hybrid architecture ablation.** A Transformer encoder pathway is implemented as an ablation control; multi-seed evidence shows that the BiLSTM-only configuration matches or exceeds the full BiLSTM–Transformer hybrid.
4. **Variance-sensitive evaluation.** Pearson correlation and Standard Deviation Ratio (StdR) are reported as co-primary diagnostics, exposing variance-collapse pathologies that directional accuracy alone cannot detect.

## Key results

On a 175-observation deduplicated, leakage-free held-out test set across 5 random seeds (`{7, 42, 123, 999, 2024}`):

| Model | DA% | Pearson r | StdR | MAE (Bt/kg/d) |
|---|---|---|---|---|
| Naive No-Change | 0.0% | 0.000 | 0.000 | 0.471 |
| Naive Random Walk | 59.6% | 0.176 | 1.000 | 0.600 |
| ARIMA(2,0,2) | 56.3% | 0.152 | 0.368 | 0.488 |
| Vanilla LSTM | 82.29 ± 0.00% | 0.398 ± 0.008 | **0.210 ± 0.007** | 0.213 ± 0.001 |
| **VMD-Augmented BiLSTM** ★ | **82.5 ± 1.8%** | **0.821 ± 0.016** | **1.091 ± 0.060** | 0.339 ± 0.023 |

The Vanilla LSTM achieves nominally lower MAE/RMSE than the proposed model, but its StdR of 0.210 sits at the variance-collapse threshold — its forecasts are dispersionally degenerate. The co-primary `(r, StdR)` diagnostic identifies this pattern.

## Repository structure

```
rubber-vmd-bilstm/
├── README.md                         ← this file
├── LICENSE                           ← MIT
├── requirements.txt                  ← Python dependencies
├── seeds.txt                         ← random seeds for multi-seed protocol
│
├── models/                           ← model definitions (PyTorch)
│   ├── bilstm_only.py                ←   proposed (deployed) model
│   ├── full_hybrid.py                ←   BiLSTM + Transformer ablation control
│   ├── transformer_only.py           ←   Transformer-only ablation control
│   └── vanilla_lstm.py               ←   unidirectional LSTM baseline (no VMD)
│
├── pipeline/                         ← data + VMD pipeline
│   ├── data_loader.py                ←   load + preprocess economic features
│   ├── expanding_window_vmd.py       ←   leakage-free VMD (Algorithm 1)
│   ├── normalization.py              ←   MinMax scaler fitted on train only
│   └── train.py                      ←   training loop (AdamW + early stopping)
│
├── evaluation/                       ← evaluation protocols
│   ├── metrics.py                    ←   DA, Pearson r, StdR, recall, MAE/RMSE
│   ├── multiseed_evaluation.py       ←   5-seed primary protocol (Table 11)
│   ├── ablation_evaluation.py        ←   architecture ablation (Table 10)
│   └── ablation_evaluation.py        ←   paired bootstrap intervals (Appendix G)
│
├── multistep/                        ← multi-horizon direct forecasting
│   └── step9d_multistep_both.py      ←   h ∈ {1,2,3,5,10,20,30}, both arch (Table 14)
│
├── analysis/                         ← supplementary analyses
│   ├── arima_grid_search.py          ←   AIC-selected ARIMA(p,0,q) (Appendix D)
│   ├── k_selection.py                ←   VMD K-sensitivity (Appendix C)
│   └── lookback_sensitivity.py       ←   L-sensitivity (Appendix F)
│
├── data_schema/                      ← data dictionary (no raw data shared)
│   ├── feature_dictionary.md         ←   24 economic features explained
│   └── sample_synthetic_data.csv     ←   100-row dummy data mirroring schema
│
└── docs/                             ← additional documentation
    ├── REPRODUCING_RESULTS.md        ←   step-by-step reproduction guide
    └── DATA_ACQUISITION.md           ←   how to obtain raw data from providers
```

## Quick start

### 1. Environment setup

```bash
# Clone repository
git clone https://github.com/talkmcp/rubber-vmd-bilstm.git
cd rubber-vmd-bilstm

# Create virtual environment (Python 3.11 recommended)
python -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify installation on synthetic data

```bash
# Run a smoke test with the included 100-row synthetic CSV
python pipeline/train.py \
    --data data_schema/sample_synthetic_data.csv \
    --model bilstm_only \
    --seed 42 \
    --max-epochs 5 \
    --smoke-test
```

If this completes without error, the environment is set up correctly. The smoke test does **not** reproduce paper results — actual training requires the full proprietary dataset (see [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md)).

### 3. Reproduce paper results (requires full data)

```bash
# Multi-seed primary results (Table 11 / Section 3.4)
python evaluation/multiseed_evaluation.py \
    --data <path-to-your-csv> \
    --model bilstm_only \
    --seeds-file seeds.txt \
    --output-dir results/multiseed/

# Architecture ablation (Table 10 / Section 3.3)
python evaluation/ablation_evaluation.py \
    --data <path-to-your-csv> \
    --seed 42 \
    --output-dir results/ablation/

# Multi-step direct forecasting (Table 14 / Section 3.6)
python multistep/step9d_multistep_both.py \
    --data <path-to-your-csv> \
    --horizons 1 2 3 5 10 20 30 \
    --seed 42 \
    --output-dir results/multistep/
```

See [`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md) for full step-by-step instructions.

## Data availability

**Raw market data are not redistributed.** Daily RSS3 FOB prices, SGX/SHFE/JPX futures settlement series, exchange rates, and Bloomberg-derived spot quotations used in the paper are licensed third-party data subject to the terms of their original providers. We cannot include them in this repository.

What **is** provided:

- **Schema and dictionary** ([`data_schema/feature_dictionary.md`](data_schema/feature_dictionary.md)) — the exact column names, units, and frequencies the pipeline expects.
- **Synthetic sample** ([`data_schema/sample_synthetic_data.csv`](data_schema/sample_synthetic_data.csv)) — 100 dummy rows for smoke-testing the pipeline.
- **Acquisition guide** ([`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md)) — sources and access procedures for each data category.

Researchers with their own provider access can reproduce the pipeline end-to-end by replacing the synthetic CSV with a real CSV matching the documented schema.

## Architecture summary

The proposed VMD-Augmented BiLSTM (≈573K parameters) uses:

- **Input**: look-back window L=30 trading days × 30 features (24 economic + 6 VMD modes)
- **Encoder**: 2-layer Bidirectional LSTM, hidden=128 per direction, dropout=0.2
- **Attention**: learnable temporal attention over the 30 look-back steps
- **Output head**: LayerNorm → Linear(256→128) → GELU → Dropout → Linear(128→64) → GELU → Linear(64→1)
- **Target**: one-step-ahead differenced price Δp_{τ+1} (Baht/kg/day)
- **Loss**: Huber (δ=0.5) — robust to outlier price shocks
- **Optimizer**: AdamW, lr=5e-4, weight-decay=1e-4, ReduceLROnPlateau ×0.5 patience 10
- **Early stopping**: 30 epochs patience, max 300 epochs

The full BiLSTM–Transformer hybrid (≈1.05M parameters, ablation-only) adds a parallel 2-layer Transformer encoder (d_model=128, 4 heads, dim_ff=512); concatenated outputs go through a fusion head. Multi-seed evidence shows this 84%-larger architecture does not improve any of the three primary metrics over BiLSTM-only.

## Multi-seed protocol

All primary results are reported as **mean ± 1 SD across 5 independent random seeds**: `{7, 42, 123, 999, 2024}` (see [`seeds.txt`](seeds.txt)). Each seed receives an independent run of training + evaluation with PyTorch deterministic mode enabled. The number of seeds is sufficient to detect variance collapse (Vanilla LSTM std(DA) = 0.00 across 5 seeds is itself a diagnostic flag) without prohibitive compute cost.

## Citation

If you use this code or framework, please cite:

```bibtex
@article{Pinitjitsamut2026MultiVMD,
  author    = {Pinitjitsamut, Montchai},
  title     = {Multi-Scale Forecasting of Natural Rubber Prices Using VMD-Augmented BiLSTM:
               A Hybrid Architecture Ablation Study},
  journal   = {Forecasting},
  year      = {2026},
  publisher = {MDPI},
  note      = {Accepted for publication}
}
```

A Zenodo DOI for this repository snapshot will be added upon journal acceptance.

## License

This code is released under the **MIT License** (see [`LICENSE`](LICENSE)). The licensed third-party market data referenced by this code are governed by their original provider terms and are **not** redistributed under this license.

## Acknowledgments

This work was conducted at the Department of Agricultural and Resource Economics, Faculty of Economics, Kasetsart University, Bangkok. During manuscript preparation the author used Claude (Anthropic) for language editing and manuscript organization. All analyses, model implementations, interpretations, and final decisions were conducted and verified by the author.

## Contact

**Montchai Pinitjitsamut**
Department of Agricultural and Resource Economics
Faculty of Economics, Kasetsart University
Bangkok 10900, Thailand
✉ montchai.p@ku.th

For questions about reproducibility, please open an issue on this repository.
