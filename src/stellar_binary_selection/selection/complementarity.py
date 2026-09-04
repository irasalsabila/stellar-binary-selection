"""Photometric–astrometric complementarity (TODO M22 — MAIN SCIENTIFIC GATE).

For every simulated binary we evaluate both detection channels and
classify it into one of four regimes:

    A: photometric ✓ / astrometric ✗
    B: photometric ✗ / astrometric ✓
    C: photometric ✓ / astrometric ✓
    D: photometric ✗ / astrometric ✗   (blind spot)

The deliverable is a calibrated map of these regimes across
(q, P, M1, d).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from .astrometric import astro_detection_probability

__all__ = [
    "REGIME_LABELS",
    "assign_regimes",
    "completeness_grid",
    "check_independence",
]


# 0 = neither, 1 = phot only, 2 = astro only, 3 = both
REGIME_LABELS = {
    0: "neither",
    1: "photometric_only",
    2: "astrometric_only",
    3: "both",
}


def assign_regimes(
    phot_detected: np.ndarray,
    astro_detected: np.ndarray,
) -> np.ndarray:
    """Integer regime label per system (see REGIME_LABELS)."""
    phot = np.asarray(phot_detected, dtype=bool)
    astro = np.asarray(astro_detected, dtype=bool)
    label = np.zeros(phot.shape, dtype=int)
    label[phot & ~astro] = 1
    label[~phot & astro] = 2
    label[phot & astro] = 3
    return label


def completeness_grid(
    df,
    *,
    bin_spec: dict[str, list[tuple[float, float]]],
    detected_col: str,
    min_per_cell: int = 20,
):
    """Compute completeness C = N_detected / N_simulated per bin.

    `bin_spec` maps a column name to a list of (lo, hi) edges. The grid
    is the Cartesian product over all specified columns.
    """
    import itertools

    import pandas as pd

    cols = list(bin_spec)
    masks = []
    for col, edges in bin_spec.items():
        masks.append([(col, lo, hi) for lo, hi in edges])

    rows = []
    for combo in itertools.product(*masks):
        sel = np.ones(len(df), dtype=bool)
        spec = {}
        for col, lo, hi in combo:
            v = pd.to_numeric(df[col], errors="coerce")
            sel &= v.between(lo, hi).fillna(False).to_numpy()
            spec[col] = (float(lo), float(hi))
        n_total = int(sel.sum())
        if n_total < min_per_cell:
            continue
        n_det = int(df.loc[sel, detected_col].sum())
        c = n_det / n_total
        # Wilson interval
        z = 1.96
        denom = 1 + z**2 / n_total
        centre = (c + z**2 / (2 * n_total)) / denom
        half = z * np.sqrt(c * (1 - c) / n_total + z**2 / (4 * n_total**2)) / denom
        row = {f"{k}_bin": v for k, v in spec.items()}
        row.update({
            "n_simulated": n_total,
            "n_detected": n_det,
            "completeness": c,
            "ci_lo": max(0.0, centre - half),
            "ci_hi": min(1.0, centre + half),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def check_independence(phot: np.ndarray, astro: np.ndarray) -> dict[str, float]:
    """Test whether the two detection channels are statistically independent.

    Under independence, P(both) = P(phot) * P(astro). We compare the
    observed joint rate against that prediction and report a chi-square
    statistic and the ratio of observed to expected.

    The PRD explicitly forbids *assuming* independence (M22.3), so this
    test must be run before combining channels.
    """
    phot = np.asarray(phot, dtype=bool)
    astro = np.asarray(astro, dtype=bool)
    n = len(phot)
    if n == 0:
        return {"n": 0}

    p_phot = phot.mean()
    p_astro = astro.mean()
    p_both_obs = (phot & astro).mean()
    p_both_exp = p_phot * p_astro

    # 2x2 contingency chi-square test of independence
    a = int((phot & astro).sum())
    b = int((phot & ~astro).sum())
    c = int((~phot & astro).sum())
    d = int((~phot & ~astro).sum())
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    denom = row1 * row2 * col1 * col2
    if denom > 0:
        chi2 = n * (a * d - b * c) ** 2 / denom
    else:
        chi2 = float("nan")

    chi2_pvalue = float(chi2_distribution.sf(chi2, 1)) if np.isfinite(chi2) else float("nan")
    return {
        "n": n,
        "p_photometric": float(p_phot),
        "p_astrometric": float(p_astro),
        "p_both_observed": float(p_both_obs),
        "p_both_expected_if_independent": float(p_both_exp),
        "ratio_observed_over_expected": float(p_both_obs / p_both_exp) if p_both_exp > 0 else float("nan"),
        "chi2_independence": float(chi2),
        "chi2_dof": 1,
        "chi2_pvalue": chi2_pvalue,
        "independent_at_5pct": bool(chi2 < 3.841) if np.isfinite(chi2) else False,
    }
