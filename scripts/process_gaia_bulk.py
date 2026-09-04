"""Process locally-stored Gaia DR3 ECSV shards into parquet files.

Use this when you already have the .csv.gz files (either manually
downloaded from ESA's bulk CDN or from a previous run of
download_gaia_bulk.py). This script does NOT touch the network.

Reads:
    data/raw/gaia_bulk/<stem>/<shard>.csv.gz
Writes:
    data/raw/gaia_bulk/<stem>.parquet       # one per table
    data/processed/gaia_nss_master.parquet  # merged four NSS tables

Run via:
    PYTHONPATH=src python scripts/process_gaia_bulk.py
    PYTHONPATH=src python scripts/process_gaia_bulk.py --only binary_masses
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


NSS_STEMS = {
    "two_body": "nss_two_body",
    "acceleration": "nss_acceleration",
    "spectro": "nss_spectro",
    "vim": "nss_vim",
}


def _nullify(value: bytes) -> bytes:
    """Replace literal Gaia 'null' tokens with empty ECSV values."""
    value = re.sub(rb'"null"', b'""', value)
    value = re.sub(rb'(?<=,|\n)null(?=,|\n|$)', b'', value)
    return value


def read_ecsv_gz(path: Path):
    """Read a Gaia-style ECSV gzip file and return a pandas DataFrame.

    Gaia ECSV uses a YAML preamble + standard CSV. astropy's strict
    ECSV reader rejects empty string conversions on float columns, so
    we strip the YAML header and parse with the python-engine pandas
    reader which handles quoted commas / null / nan correctly.
    """
    import gzip
    import re
    from io import StringIO

    import pandas as pd

    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8", errors="replace")

    # Strip YAML preamble (# ...).
    lines = text.split("\n")
    data_start = next(
        (i for i, l in enumerate(lines) if not l.startswith("#") and l.strip()),
        None,
    )
    if data_start is None:
        raise ValueError(f"no data section found in {path}")
    header_line = lines[data_start]
    header = [h.strip() for h in header_line.split(",")]
    body = "\n".join(lines[data_start + 1 :])

    # Replace Gaia 'null' tokens with ECSV/pandas-standard empty.
    body = re.sub(r'"null"', '""', body)
    body = re.sub(r"(?<=,|\n)null(?=,|\n|$)", "", body, flags=re.MULTILINE)

    df = pd.read_csv(
        StringIO(body),
        header=None,
        names=header,
        na_values=["", "NaN", "nan", "NULL"],
        engine="python",
    )

    # Coerce object columns to numeric where possible (keep string / list
    # columns like nss_solution_type and corr_vec as object).
    preserve_object = {"nss_solution_type", "corr_vec", "thiele_innes_*"}
    for c in df.columns:
        if df[c].dtype == object and c not in preserve_object:
            # Skip columns whose name contains brackets / array hints.
            if c.startswith("[") or c.endswith("]"):
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def process_one(stem_dir: Path, out_parquet: Path) -> int:
    shards = sorted(stem_dir.glob("*.csv.gz"))
    if not shards:
        print(f"  [skip] {stem_dir}: no .csv.gz files")
        return 0
    print(f"  {stem_dir.name}: reading {len(shards)} shard(s)...", flush=True)
    frames = []
    for s in tqdm(shards, desc=f"  {stem_dir.name}", unit="file"):
        frames.append(read_ecsv_gz(s))
    full = __import__("pandas").concat(frames, ignore_index=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_parquet, index=False)
    print(f"    → {out_parquet} ({len(full):,} rows × {len(full.columns)} cols)", flush=True)
    return int(len(full))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bulk-dir",
        default="data/raw/gaia_bulk",
        help="Directory containing per-table subfolders with .csv.gz shards.",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(NSS_STEMS) + ["binary_masses", "all", "nss"],
        default=["all"],
    )
    args = p.parse_args()

    bulk_dir = REPO_ROOT / args.bulk_dir

    targets: list[str] = []
    for choice in args.only:
        if choice == "all":
            targets = list(NSS_STEMS) + ["binary_masses"]
            break
        if choice == "nss":
            targets += list(NSS_STEMS)
        else:
            targets.append(choice)

    nss_rows_total = 0
    for label in targets:
        stem_dir_name = "binary_masses" if label == "binary_masses" else NSS_STEMS[label]
        stem_dir = bulk_dir / stem_dir_name
        out_parquet = bulk_dir / f"{stem_dir_name}.parquet"
        n = process_one(stem_dir, out_parquet)
        if label in NSS_STEMS:
            nss_rows_total += n

    if any(l in NSS_STEMS for l in targets):
        master_out = REPO_ROOT / "data" / "processed" / "gaia_nss_master.parquet"
        print("\n=== building gaia_nss_master.parquet ===", flush=True)
        import pandas as pd

        frames = []
        for label, stem_dir_name in NSS_STEMS.items():
            path = bulk_dir / f"{stem_dir_name}.parquet"
            if not path.exists():
                print(f"  skipping {label} ({stem_dir_name}.parquet not present)")
                continue
            df = pd.read_parquet(path)
            df["nss_table_origin"] = label
            frames.append(df)
        if frames:
            full = pd.concat(frames, ignore_index=True)
            master_out.parent.mkdir(parents=True, exist_ok=True)
            full.to_parquet(master_out, index=False)
            print(f"  → {master_out} ({len(full):,} rows × {len(full.columns)} cols)")
        else:
            print("  no NSS parquets available to merge")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())