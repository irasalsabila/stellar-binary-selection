"""Selection function and complementarity evaluation (TODO M17–M22)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.evaluation.metrics import binomial_completeness
from stellar_binary_selection.utils import write_run_metadata


def select_q_mass_distance_grid(df: pd.DataFrame, q_grid, m_grid, d_grid):
    for q_lo, q_hi in q_grid:
        for m_lo, m_hi in m_grid:
            for d_lo, d_hi in d_grid:
                mask = (
                    df["is_binary"]
                    & df["q"].between(q_lo, q_hi)
                    & df["m1_msun"].between(m_lo, m_hi)
                    & df["distance_pc"].between(d_lo, d_hi)
                )
                yield (q_lo, q_hi), (m_lo, m_hi), (d_lo, d_hi), mask


def compute_c_phot(df: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    """Bin synthetic systems in (q, M1, d) and compute C_phot."""
    q_grid = [(q, q + 0.1) for q in np.arange(0.1, 1.05, 0.1)]
    m_grid = [(0.6, 0.9), (0.9, 1.2), (1.2, 1.4)]
    d_grid = [(20, 60), (60, 100), (100, 150), (150, 200)]

    cells = list(select_q_mass_distance_grid(df, q_grid, m_grid, d_grid))
    print(f"  computing completeness for {len(cells)} (q, M_1, d) cells...", flush=True)

    rows = []
    score = df["p_binary_score"].to_numpy() if "p_binary_score" in df.columns else df["apparent_G"].to_numpy()
    # Use overluminosity proxy: ΔG < 0 for now (until real model scores exist)
    proxy_detected = score < np.nanpercentile(score, 5)
    for (q_lo, q_hi), (m_lo, m_hi), (d_lo, d_hi), mask in tqdm(cells, desc="C_phot cells"):
        n_total = int(mask.sum())
        if n_total < 20:
            continue
        detected = mask & proxy_detected
        n_det = int(detected.sum())
        c, lo, hi = binomial_completeness(n_det, n_total)
        rows.append({"q_bin": (q_lo, q_hi), "m_bin": (m_lo, m_hi), "d_bin": (d_lo, d_hi), "C_phot": c, "C_phot_lo": lo, "C_phot_hi": hi, "n": n_total})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["selection_phot", "complementarity", "all"], default="all")
    p.add_argument("--synthetic", default="data/synthetic/synthetic_v1.parquet")
    p.add_argument("--metadata-dir", default="results/runs/evaluate")
    args = p.parse_args()

    df = pd.read_parquet(REPO_ROOT / args.synthetic)
    print(f"Loaded {len(df):,} synthetic systems for evaluation", flush=True)

    if "p_binary_score" not in df.columns:
        # placeholder score = brightness residual (overluminous binary => brighter in G)
        df["p_binary_score"] = df["apparent_G"] - df["apparent_G"].mean()

    out = REPO_ROOT / "results" / "tables" / "c_phot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = compute_c_phot(df, score_threshold=0.5)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} completeness cells to {out.relative_to(REPO_ROOT)}")

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={"task": args.task},
        metrics={"n_rows": len(df), "n_cells": len(table)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())