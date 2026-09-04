"""Photometric selection-function utilities (TODO M20)."""

from __future__ import annotations

import numpy as np

__all__ = ["photometric_threshold", "calibrate_overluminosity_threshold"]


def photometric_threshold(apparent_G: np.ndarray, control_apparent_G: np.ndarray, *, fpr: float = 0.01) -> float:
    """Magnitude threshold on the overluminosity score giving the requested
    FPR against an apparently-single control sample."""
    return calibrate_overluminosity_threshold(control_apparent_G, fpr=fpr)


def calibrate_overluminosity_threshold(single_scores: np.ndarray, *, fpr: float = 0.01) -> float:
    """Threshold on an overluminosity score calibrated on single stars.

    The score is oriented so that LARGER values mean MORE overluminous
    (i.e. brighter than the single-star expectation). The threshold is
    the (1 - fpr) quantile of the single-star score distribution, so a
    fraction `fpr` of true singles are (falsely) flagged.
    """
    scores = np.asarray(single_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("no finite single-star scores to calibrate against")
    return float(np.quantile(scores, 1.0 - fpr))