"""Leakage audit: ensure forbidden binary diagnostics never enter model inputs (TODO M28)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


FORBIDDEN = [
    "ruwe",
    "non_single_star",
    "nss_solution_type",
    "ipd_frac_multi_peak",
    "ipd_gof_harmonic_amplitude",
    "m1",
    "m2",
    "fluxratio",
    "combination_method",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", default="data/synthetic/synthetic_v1.parquet")
    args = p.parse_args()

    import pandas as pd
    df = pd.read_parquet(REPO_ROOT / args.synthetic)
    cols = set(df.columns)
    leaks = [c for c in FORBIDDEN if c in cols]
    if leaks:
        print(f"FAIL: forbidden columns present in {args.synthetic}: {leaks}")
        return 1
    print(f"OK: no forbidden columns in {args.synthetic} (n={len(df):,}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())