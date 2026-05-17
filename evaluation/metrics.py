"""
evaluation/metrics.py

Evaluation metrics for differenced commodity price forecasts.
All co-primary metrics from Pinitjitsamut (2026), Section 2.5 / Table 7.

The two co-primary metrics — Pearson r and StdR — are reported jointly:
neither alone is sufficient. StdR below 0.20 is flagged as the
variance-collapse threshold (paper Section 3.1).
"""
from __future__ import annotations
import numpy as np


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predicted and actual differenced series."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def stdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard Deviation Ratio: std(pred) / std(actual). Ideal = 1.0.

    Below 0.20 indicates variance collapse — the model has converged
    to a near-flat forecast and is dispersionally degenerate.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    s_actual = np.std(y_true, ddof=1)
    if s_actual == 0:
        return float("nan")
    return float(np.std(y_pred, ddof=1) / s_actual)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional accuracy on the NON-ZERO subset of the actual series.
    Following paper Section 2.5 / Table 7: days where Δp_t = 0 are
    excluded from the denominator.

    Returns percentage in [0, 100].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        return float("nan")
    correct = (np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])).sum()
    return 100.0 * correct / nonzero.sum()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error on differenced series (Baht/kg/day)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error on differenced series (Baht/kg/day)."""
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(err ** 2)))


def class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Up-recall = TP / (TP + FN). Down-recall = TN / (TN + FP).
    Computed on the non-zero subset.

    Returns dict with keys 'up_recall', 'down_recall' (both in [0, 100]).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    nonzero = y_true != 0

    actual_up = (y_true[nonzero] > 0)
    actual_dn = (y_true[nonzero] < 0)
    pred_up = (y_pred[nonzero] > 0)
    pred_dn = (y_pred[nonzero] < 0)

    tp = (actual_up & pred_up).sum()
    fn = (actual_up & pred_dn).sum()
    tn = (actual_dn & pred_dn).sum()
    fp = (actual_dn & pred_up).sum()

    return {
        "up_recall":   100.0 * tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "down_recall": 100.0 * tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    }


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the full metric panel reported in the paper."""
    out = {
        "DA_nonzero":  directional_accuracy(y_true, y_pred),
        "pearson_r":   pearson_r(y_true, y_pred),
        "stdr":        stdr(y_true, y_pred),
        "mae":         mae(y_true, y_pred),
        "rmse":        rmse(y_true, y_pred),
    }
    out.update(class_recall(y_true, y_pred))
    return out


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.5, 175)
    y_good = y + rng.normal(0, 0.2, 175)              # high r, near-1 StdR
    y_collapsed = 0.2 * np.sign(y)                    # variance collapse pattern

    print("─── Good forecast ───")
    for k, v in all_metrics(y, y_good).items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n─── Variance-collapsed forecast ───")
    for k, v in all_metrics(y, y_collapsed).items():
        print(f"  {k:12s}: {v:.4f}")
