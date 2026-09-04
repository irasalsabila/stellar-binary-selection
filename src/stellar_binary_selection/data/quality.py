"""Real Gaia data quality helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["basic_validation"]


def basic_validation(df: pd.DataFrame) -> dict[str, int]:
    """Lightweight M4.1-style validation summary."""
    return {
        "n_rows": int(len(df)),
        "n_unique_source_id": int(df["source_id"].nunique()),
        "n_missing_ruwe": int(df["ruwe"].isna().sum()),
        "n_missing_g_mag": int(df["phot_g_mean_mag"].isna().sum()),
        "n_negative_parallax": int((df["parallax"] < 0).sum()),
        "n_inf_parallax": int(np.isinf(df["parallax"].to_numpy()).sum()),
        "n_low_ruwe": int((df["ruwe"] < 1.4).sum()),
        "n_high_ruwe": int((df["ruwe"] >= 1.4).sum()),
        "n_non_single_star": int((df["non_single_star"] > 0).sum()),
    }