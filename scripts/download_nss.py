"""Download Gaia DR3 NSS and binary_masses validation sets via ADQL.

This is the ADQL path — for most users the bulk CDN downloader
(`scripts/download_gaia_bulk.py`) is faster and avoids putting NSS
table scans through the TAP queue. Use this script only if you want
to apply custom ADQL filters (e.g. restrict to a sky region).

Each NSS sub-table is downloaded separately to preserve the fact that
a given source_id can legitimately have multiple solutions across or
within tables. The four outputs are merged locally into
data/processed/gaia_nss_master.parquet.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.utils import hash_file, write_run_metadata


QUERIES = {
    "nss_two_body": ("queries/gaia_nss_two_body.adql", "gaia_dr3_nss_two_body.parquet"),
    "nss_acceleration": ("queries/gaia_nss_acceleration.adql", "gaia_dr3_nss_acceleration.parquet"),
    "nss_spectro": ("queries/gaia_nss_spectro.adql", "gaia_dr3_nss_spectro.parquet"),
    "nss_vim": ("queries/gaia_nss_vim.adql", "gaia_dr3_nss_vim.parquet"),
    "binary_masses": ("queries/gaia_binary_masses.adql", "gaia_dr3_binary_masses.parquet"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--metadata-dir", default="results/runs/gaia_validation_download")
    parser.add_argument("--only", nargs="+", choices=list(QUERIES) + ["nss", "all"], default=["all"])
    return parser.parse_args()


def stage_query(adql_path: Path, output: Path) -> int:
    """Async TAP job + polling progress; results saved via pandas."""
    from astroquery.gaia import Gaia

    scratch_dir = REPO_ROOT / "data" / "raw" / "_vot_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    vot_path = scratch_dir / f"{output.stem}-result.vot.gz"

    print(f"  Submitting async query for {adql_path.name}...", flush=True)
    job = Gaia.launch_job_async(adql_path.read_text(), dump_to_file=True, output_file=str(vot_path))
    jobid = getattr(job, "jobid", "?")
    print(f"  Job id={jobid}. Waiting for Gaia Archive...", flush=True)

    poll_seconds = 10
    final_phase: str = ""
    try:
        with tqdm(total=0, desc=f"Gaia {jobid[:8]}", unit="poll",
                  bar_format="{desc} [{elapsed}] {postfix}") as bar:
            while True:
                try:
                    phase = job.get_phase(update=True)
                except Exception as exc:  # noqa: BLE001
                    phase = "UNKNOWN"
                    bar.set_postfix_str(f"phase={phase} (poll error: {exc})")
                else:
                    bar.set_postfix_str(f"phase={phase}")
                bar.update(0)
                final_phase = phase
                if phase in {"COMPLETED", "ERROR", "ABORTED"}:
                    break
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nCancelled. The server-side job will expire automatically.")
        try:
            job.abort()
        except Exception:
            pass
        raise

    if getattr(job, "failed", False) or final_phase == "ERROR":
        msg = job.get_error_messages() if hasattr(job, "get_error_messages") else job.failed
        raise SystemExit(f"Gaia async job failed: {msg}")

    print(f"\n  Query completed (phase={final_phase}). Loading results...", flush=True)
    table = job.get_results()
    n = len(table)
    if n == 0:
        print(f"  WARN: {adql_path.name} returned 0 rows.")
        return 0
    print(f"  {n:,} rows × {len(table.columns)} columns", flush=True)

    df = table.to_pandas()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tqdm(total=2, desc="parquet write", unit="step",
                  bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}] {postfix}") as bar:
            df.to_parquet(tmp_path, index=False)
            bar.set_postfix_str("rows → bytes")
            bar.update(1)
            shutil.move(str(tmp_path), output)
            bar.set_postfix_str(f"moved → {output.name}")
            bar.update(1)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return int(len(df))


def merge_nss_tables(raw_dir: Path, out_path: Path) -> int:
    """Build data/processed/gaia_nss_master.parquet from the four NSS tables."""
    import pandas as pd

    nss_stems = [
        ("two_body", "gaia_dr3_nss_two_body.parquet"),
        ("acceleration", "gaia_dr3_nss_acceleration.parquet"),
        ("spectro", "gaia_dr3_nss_spectro.parquet"),
        ("vim", "gaia_dr3_nss_vim.parquet"),
    ]
    frames = []
    for label, fname in nss_stems:
        path = raw_dir / fname
        if not path.exists():
            print(f"  skipping {label} ({fname} not present)")
            continue
        df = pd.read_parquet(path)
        df["nss_table_origin"] = label
        frames.append(df)
    if not frames:
        return 0
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"  Wrote {len(full):,} rows × {len(full.columns)} cols → {out_path}")
    return int(len(full))


def main() -> int:
    args = parse_args()
    metadata: dict = {}

    raw_dir = REPO_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    targets: list[str] = []
    for choice in args.only:
        if choice == "all":
            targets = list(QUERIES)
            break
        if choice == "nss":
            targets += ["nss_two_body", "nss_acceleration", "nss_spectro", "nss_vim"]
        else:
            targets.append(choice)

    seen: set[str] = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]

    for key in targets:
        adql_rel, fname = QUERIES[key]
        adql_path = REPO_ROOT / adql_rel
        output = raw_dir / fname
        if output.exists() and not args.force:
            print(f"[skip] {output.relative_to(REPO_ROOT)} already exists (pass --force).")
            metadata[key] = {"status": "skipped", "path": str(output)}
            continue
        print(f"\n=== {key} ===")
        n = stage_query(adql_path, output)
        checksum = hash_file(output) if n else None
        metadata[key] = {"status": "ok" if n else "empty", "n_rows": n, "sha256": checksum}
        print(f"  → {n:,} rows", flush=True)

    if any(t.startswith("nss_") for t in targets):
        master_out = REPO_ROOT / "data" / "processed" / "gaia_nss_master.parquet"
        print("\n=== building gaia_nss_master.parquet ===", flush=True)
        n = merge_nss_tables(raw_dir, master_out)
        metadata["nss_master"] = {"n_rows": n, "path": str(master_out)}

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={"queries": list(QUERIES), "only": args.only},
        metrics=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())