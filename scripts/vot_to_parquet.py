"""Convert a Gaia Archive VOTable (optionally gzip-compressed) to parquet.

Useful when you downloaded the results from the Gaia web UI instead of
running the async TAP scripts.

Run via:
    PYTHONPATH=src python scripts/vot_to_parquet.py data/raw/gaia_dr3_nss.vot.gz
    PYTHONPATH=src python scripts/vot_to_parquet.py data/raw/*.vot.gz --out data/raw/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="VOTable file(s) (.vot or .vot.gz).")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: same directory as each input, with .parquet extension.",
    )
    return p.parse_args()


def convert_one(input_path: Path, out_dir: Path) -> Path:
    import pandas as pd
    from astropy.io import votable as avotable
    from astropy.table import Table

    print(f"Reading {input_path}...", flush=True)
    if input_path.suffix == ".gz":
        inner = input_path.with_suffix("")  # strip .gz → .vot
    else:
        inner = input_path
    # astropy handles .gz transparently
    vot = avotable.parse_single_table(str(input_path)).to_table()
    df = vot.to_pandas()

    out_path = out_dir / (inner.stem + ".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  → {out_path}  ({len(df):,} rows × {len(df.columns)} cols)", flush=True)
    return out_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out) if args.out else None
    for inp in args.inputs:
        inp_path = Path(inp)
        target = out_dir if out_dir else inp_path.parent
        convert_one(inp_path, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())