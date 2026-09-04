"""Download the immutable Gaia DR3 + 2MASS parent catalogue.

Run via:
    python scripts/download_gaia.py            # full parent sample
    python scripts/download_gaia.py --test     # 1,000-row smoke test
    python scripts/download_gaia.py --rows N   # custom TOP N (default 250000)

This implements TODO M2.2–M2.4. The output file is treated as immutable:
subsequent runs overwrite nothing — the script refuses to run if the
target path already exists unless `--force` is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.utils import hash_file, utc_now_iso, write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adql",
        default="queries/gaia_2mass_parent.adql",
        help="ADQL file to execute.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/gaia_dr3_2mass_parent.parquet",
        help="Output immutable parquet file.",
    )
    parser.add_argument("--test", action="store_true", help="Run the 1k-row test query.")
    parser.add_argument("--rows", type=int, default=250_000, help="TOP N for the parent query.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing raw parquet (DANGEROUS — only for refresh).",
    )
    parser.add_argument(
        "--metadata-dir",
        default="results/runs/gaia_parent_download",
        help="Where to save the run metadata snapshot.",
    )
    return parser.parse_args()


def resolve_query(adql_path: Path, rows: int, test: bool) -> str:
    """Load ADQL and (optionally) rewrite TOP N."""
    text = adql_path.read_text()
    if test:
        return text.replace("SELECT TOP 250000", "SELECT TOP 1000", 1)
    return text.replace("SELECT TOP 250000", f"SELECT TOP {rows}", 1)


def ensure_immutable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite existing immutable file: {path}\n"
            "Re-run with --force only if you intend to refresh the dataset."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def stage_download(adql: str, output: Path) -> int:
    """Execute the ADQL via the async TAP interface and stream to parquet.

    Synchronous jobs time out for the full ~250k parent-sample query, so
    we launch async and poll until completion (PRD §M2.4). For large
    results we use `dump_to_file=True` so the table is materialised on
    disk and only read row-by-row when written to parquet.
    """
    import time

    import pandas as pd
    from tqdm import tqdm

    from astroquery.gaia import Gaia

    print("Submitting async query to Gaia Archive...", flush=True)
    scratch_dir = REPO_ROOT / "data" / "raw" / "_vot_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    vot_path = scratch_dir / "gaia_async_result.vot.gz"

    job = Gaia.launch_job_async(adql, dump_to_file=True, output_file=str(vot_path))
    jobid = getattr(job, "jobid", "?")
    print(f"Job submitted (id={jobid}). Waiting for Gaia Archive...", flush=True)

    poll_seconds = 10
    final_phase: str = ""
    try:
        # Indeterminate bar — the Archive does not stream partial row
        # counts, so this shows elapsed time + current phase instead.
        with tqdm(total=0, desc=f"Gaia {jobid[:8]}", unit="poll", bar_format="{desc} [{elapsed}] {postfix}") as bar:
            while True:
                try:
                    phase = job.get_phase(update=True)
                except Exception as exc:  # noqa: BLE001 — transient TAP errors
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
        print("\nCancelled by user. The server-side job will expire automatically.")
        try:
            job.abort()
        except Exception:
            pass
        raise

    if getattr(job, "failed", False) or final_phase == "ERROR":
        msg = job.get_error_messages() if hasattr(job, "get_error_messages") else job.failed
        raise SystemExit(f"Gaia async job failed: {msg}")

    print(f"\nQuery completed (phase={final_phase}). Downloading results to parquet...", flush=True)

    # job.get_results() handles both in-memory and file-dumped results.
    print("  loading VOTable → astropy table...", flush=True)
    table = job.get_results()
    n = len(table)
    if n == 0:
        raise SystemExit("ADQL query returned 0 rows; aborting.")
    print(f"  {n:,} rows × {len(table.columns)} columns", flush=True)

    # astropy.table.write(format='parquet') misinterprets some object-typed
    # columns. Convert via pandas first for robustness across Gaia
    # schema revisions.
    print("  converting to pandas DataFrame...", flush=True)
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


def main() -> int:
    args = parse_args()
    output = REPO_ROOT / args.output
    ensure_immutable(output, force=args.force)

    adql_path = REPO_ROOT / args.adql
    if args.test:
        adql_path = REPO_ROOT / "queries" / "gaia_2mass_parent_test.adql"

    adql = resolve_query(adql_path, rows=args.rows, test=args.test)
    print(f"Executing ADQL from {adql_path} → {output}")

    n_rows = stage_download(adql, output)
    checksum = hash_file(output)
    print(f"Wrote {n_rows} rows to {output} (sha256={checksum[:16]}…)")

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={
            "adql_file": str(adql_path.relative_to(REPO_ROOT)),
            "output_file": str(output.relative_to(REPO_ROOT)),
            "rows_requested": args.rows if not args.test else 1000,
        },
        metrics={"n_rows": n_rows, "sha256": checksum, "timestamp_utc": utc_now_iso()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())