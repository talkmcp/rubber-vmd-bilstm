"""
paired_bootstrap_ci.py
=======================

Compute non-parametric paired bootstrap 95% confidence intervals for the
co-primary diagnostics (Pearson r, Standard Deviation Ratio, paired r-difference,
paired StdR-difference) at the deployment horizon h = 1, on the leakage-free
deduplicated test partition (n = 175).

This script is designed to be self-contained: provide it with three numpy arrays
or 1-D CSV files of length 175 (actuals, predictions from VMD-Augmented BiLSTM,
predictions from Vanilla LSTM), and it prints out the bootstrap CIs that should
be inserted into Table G.2 in Appendix G.

REQUIRED INPUTS
---------------
- actual.csv               : 175 rows, 1 column, header optional. Realised
                             first-differenced RSS3 FOBm1 (Baht/kg/day).
- pred_proposed.csv        : 175 rows, predictions from VMD-Augmented BiLSTM
                             at seed 42 (matching the seed-42 row of Table 12).
- pred_vanilla.csv         : 175 rows, predictions from Vanilla LSTM at seed 42.

USAGE
-----
    python paired_bootstrap_ci.py actual.csv pred_proposed.csv pred_vanilla.csv

Adjust B (number of bootstrap resamples) at the top if needed. Default 10,000
is appropriate for a 95% CI; 1,000 is faster for sanity checking.

The block-bootstrap option below preserves serial dependence if the residuals
exhibit autocorrelation; for the deduplicated differenced series we use the
ordinary i.i.d. paired bootstrap on observation indices, which is appropriate
for forecast-error inference on a differenced (approximately uncorrelated)
target.
"""
import sys
import numpy as np
import pandas as pd

# ---- Configuration ---------------------------------------------------
B = 10_000              # bootstrap resamples
SEED = 42               # bootstrap RNG seed (NOT model seed)
CI_LEVEL = 0.95         # two-sided level

# ---- Statistics ------------------------------------------------------
def pearson_r(y, yhat):
    return np.corrcoef(y, yhat)[0, 1]

def stdr(y, yhat):
    return np.std(yhat, ddof=1) / np.std(y, ddof=1)

# ---- Bootstrap engine ------------------------------------------------
def paired_bootstrap(y, p1, p2, B=B, rng=None):
    """
    i.i.d. paired bootstrap over observation indices.
    Returns dict of arrays of bootstrap-replicated statistics.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(y)
    out = {
        "r1": np.empty(B), "r2": np.empty(B),
        "stdr1": np.empty(B), "stdr2": np.empty(B),
        "dr": np.empty(B), "dstdr": np.empty(B),
    }
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yb, p1b, p2b = y[idx], p1[idx], p2[idx]
        out["r1"][b] = pearson_r(yb, p1b)
        out["r2"][b] = pearson_r(yb, p2b)
        out["stdr1"][b] = stdr(yb, p1b)
        out["stdr2"][b] = stdr(yb, p2b)
    out["dr"] = out["r1"] - out["r2"]
    out["dstdr"] = out["stdr1"] - out["stdr2"]
    return out

def ci(arr, level=CI_LEVEL):
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(arr, [alpha, 1 - alpha])
    return float(lo), float(hi)

# ---- Main ------------------------------------------------------------
def load1d(path):
    df = pd.read_csv(path, header=None)
    arr = df.iloc[:, 0].to_numpy()
    # If first row was a header that pandas read as data, drop it
    if isinstance(arr[0], str):
        arr = df.iloc[1:, 0].to_numpy(dtype=float)
    return arr.astype(float)

def main():
    if len(sys.argv) != 4:
        print("Usage: python paired_bootstrap_ci.py <actual.csv> <pred_proposed.csv> <pred_vanilla.csv>")
        sys.exit(1)
    y  = load1d(sys.argv[1])
    p1 = load1d(sys.argv[2])
    p2 = load1d(sys.argv[3])
    assert len(y) == len(p1) == len(p2), \
        f"Length mismatch: y={len(y)}, p1={len(p1)}, p2={len(p2)}"
    n = len(y)
    print(f"\nLoaded n = {n} observations.")
    print(f"Point estimates:")
    print(f"  Proposed: r = {pearson_r(y, p1):.4f}, StdR = {stdr(y, p1):.4f}")
    print(f"  Vanilla : r = {pearson_r(y, p2):.4f}, StdR = {stdr(y, p2):.4f}")
    print(f"  Differences (Proposed - Vanilla):")
    print(f"    Δr     = {pearson_r(y,p1) - pearson_r(y,p2):.4f}")
    print(f"    ΔStdR  = {stdr(y,p1)    - stdr(y,p2):.4f}")

    print(f"\nRunning paired bootstrap with B = {B} ...")
    boots = paired_bootstrap(y, p1, p2)

    print(f"\n{'='*60}")
    print(f"Paired Bootstrap 95% CIs (n = {n}, B = {B}, seed = {SEED})")
    print(f"{'='*60}")
    print(f"{'Quantity':<32}{'Point':>10}{'2.5%':>10}{'97.5%':>10}")
    print(f"{'-'*60}")
    rows = [
        ("Proposed Pearson r",   pearson_r(y, p1), ci(boots["r1"])),
        ("Vanilla  Pearson r",   pearson_r(y, p2), ci(boots["r2"])),
        ("Proposed StdR",        stdr(y, p1),      ci(boots["stdr1"])),
        ("Vanilla  StdR",        stdr(y, p2),      ci(boots["stdr2"])),
        ("Δr (Proposed − Vanilla)",
            pearson_r(y, p1) - pearson_r(y, p2),
            ci(boots["dr"])),
        ("ΔStdR (Proposed − Vanilla)",
            stdr(y, p1)    - stdr(y, p2),
            ci(boots["dstdr"])),
    ]
    for label, pt, (lo, hi) in rows:
        print(f"{label:<32}{pt:>10.4f}{lo:>10.4f}{hi:>10.4f}")

    # Inferential summary
    lo_dr, hi_dr = ci(boots["dr"])
    excludes_zero_dr = (lo_dr > 0) or (hi_dr < 0)
    lo_dstdr, hi_dstdr = ci(boots["dstdr"])
    excludes_zero_dstdr = (lo_dstdr > 0) or (hi_dstdr < 0)
    print(f"\nInferential summary:")
    print(f"  Δr 95% CI excludes 0:    {excludes_zero_dr}  ({lo_dr:.4f}, {hi_dr:.4f})")
    print(f"  ΔStdR 95% CI excludes 0: {excludes_zero_dstdr}  ({lo_dstdr:.4f}, {hi_dstdr:.4f})")

if __name__ == "__main__":
    main()
