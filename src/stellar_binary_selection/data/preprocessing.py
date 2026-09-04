"""Derived observational quantities (M4.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_derived_observables"]


def add_derived_observables(df: pd.DataFrame) -> pd.DataFrame:
    """Add colour indices and preliminary absolute magnitudes."""
    out = df.copy()
    out["bp_rp"] = out["phot_bp_mean_mag"] - out["phot_rp_mean_mag"]
    out["g_j"] = out["phot_g_mean_mag"] - out["j_m"]
    out["g_ks"] = out["phot_g_mean_mag"] - out["ks_m"]
    out["j_h"] = out["j_m"] - out["h_m"]
    out["h_ks"] = out["h_m"] - out["ks_m"]
    out["j_ks"] = out["j_m"] - out["ks_m"]
    parallax = out["parallax"].clip(lower=0.1)
    out["distance_pc_prelim"] = 1000.0 / parallax
    out["m_g_prelim"] = (
        out["phot_g_mean_mag"] - 5.0 * np.log10(out["distance_pc_prelim"]) + 5.0
    )
    return out