"""Classical main-sequence ridge overluminosity baseline (TODO M12)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["MainSequenceRidgeBaseline"]


class MainSequenceRidgeBaseline(BaseEstimator, ClassifierMixin):
    """Polynomial ridge M_G(BP-RP) — overluminosity criterion.

    The classifier is fit on an apparently-single training set
    (Sample B from M5). At evaluation time, score = (M_G_observed -
    M_G_ridge) / sigma(M_G); values more negative = more luminous
    than the single-star ridge = binary candidate.

    Uses naive 1000/parallax distances internally — only suitable for
    the B0 baseline, not the final PI-NN pipeline.
    """

    def __init__(self, polynomial_order: int = 3, threshold_factor: float = 1.5):
        self.polynomial_order = polynomial_order
        self.threshold_factor = threshold_factor
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):  # noqa: D401
        bp_rp = X[:, 0]
        m_g = X[:, 1]
        order = self.polynomial_order
        A = np.vander(bp_rp, order + 1, increasing=True)
        self.coef_, *_ = np.linalg.lstsq(A, m_g, rcond=None)
        residual = m_g - A @ self.coef_
        self.sigma_residual_ = float(np.std(residual))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        bp_rp = X[:, 0]
        m_g = X[:, 1]
        order = self.polynomial_order
        A = np.vander(bp_rp, order + 1, increasing=True)
        ridge = A @ self.coef_
        return (ridge - m_g) / self.sigma_residual_  # higher = more overluminous

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > self.threshold_factor).astype(int)