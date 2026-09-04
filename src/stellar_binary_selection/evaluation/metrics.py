"""Evaluation metrics and stratification utilities (TODO M17, M26)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

__all__ = [
    "binary_classification_metrics",
    "regression_metrics",
    "binomial_completeness",
]


def binary_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision_at_1pct_fpr": _precision_at_fpr(y_true, y_score, 0.01),
        "recall_at_1pct_fpr": _recall_at_fpr(y_true, y_score, 0.01),
        "precision_at_5pct_fpr": _precision_at_fpr(y_true, y_score, 0.05),
        "recall_at_5pct_fpr": _recall_at_fpr(y_true, y_score, 0.05),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_bias": float(np.median(err)),
    }


def binomial_completeness(n_detected: int, n_total: int, *, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson-style binomial completeness interval."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_detected / n_total
    denom = 1.0 + z ** 2 / n_total
    centre = (p + z ** 2 / (2.0 * n_total)) / denom
    halfw = (z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4.0 * n_total ** 2))) / denom
    return p, max(0.0, centre - halfw), min(1.0, centre + halfw)


def _precision_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    neg = (y_true == 0).sum()
    target_neg = int(round(target_fpr * neg))
    tp, fp = 0, 0
    last = 0.0
    for i in range(len(s_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            if fp == target_neg:
                last = tp / max(tp + fp, 1)
                break
    return last


def _recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    neg = (y_true == 0).sum()
    pos = (y_true == 1).sum()
    target_neg = int(round(target_fpr * neg))
    tp, fp = 0, 0
    last = 0.0
    for i in range(len(s_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            if fp == target_neg:
                last = tp / max(pos, 1)
                break
    return last