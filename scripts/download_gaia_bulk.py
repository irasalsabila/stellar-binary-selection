"""Bulk-download Gaia DR3 NSS and binary_masses tables from ESA's CDN.

The CDN's directory URLs (`https://cdn.gea.esac.esa.int/Gaia/...`) redirect
to ESA's JS file browser which is not script-friendly. The correct
listing endpoint is the underlying CDN77 S3-compatible origin:

    https://gaia.eu-1.cdn77-storage.com/?prefix=<PREFIX>&delimiter=/

which returns XML of the form:

    <Contents><Key>Gaia/gdr3/.../NssTwoBodyOrbit_1.csv.gz</Key>...</Contents>

Actual files are then fetched from `https://cdn.gea.esac.esa.int/<Key>`.

This script:
- lists shards via the cdn77-storage origin,
- parallel-downloads them with a progress bar per shard,
- merges per-table shards into data/raw/gaia_bulk/<table>.parquet,
- merges the four NSS tables into data/processed/gaia_nss_master.parquet,
- verifies MD5 checksums against the published _MD5SUM.txt when available,
- hard-fails on zero shards so a bad URL cannot silently produce empty tables.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


LIST_BASE = "https://gaia.eu-1.cdn77-storage.com"
DOWNLOAD_BASE = "https://cdn.gea.esac.esa.int"


# Table name → (CDN prefix, output stem)
TABLES = {
    "nss_two_body_orbit": ("Gaia/gdr3/Non-single_stars/nss_two_body_orbit/", "nss_two_body"),
    "nss_acceleration_astro": ("Gaia/gdr3/Non-single_stars/nss_acceleration_astro/", "nss_acceleration"),
    "nss_non_linear_spectro": ("Gaia/gdr3/Non-single_stars/nss_non_linear_spectro/", "nss_spectro"),
    "nss_vim_fl": ("Gaia/gdr3/Non-single_stars/nss_vim_fl/", "nss_vim"),
    "binary_masses": ("Gaia/gdr3/Performance_verification/binary_masses/", "binary_masses"),
}


# ESA bulk-data CSV columns for each NSS table are uniformly .csv.gz.
SHARD_SUFFIXES = (".csv.gz", ".fits.gz", ".vot.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(TABLES) + ["nss", "all"],
        default=["all"],
        help="Subset of tables to download.",
    )
    p.add_argument(
        "--raw-dir",
        default="data/raw/gaia_bulk",
        help="Where to store the individual shard files.",
    )
    p.add_argument("--workers", type=int, default=4, help="Parallel HTTP threads.")
    p.add_argument("--force", action="store_true", help="Re-download shards already on disk.")
    p.add_argument(
        "--skip-md5",
        action="store_true",
        help="Do not verify the published _MD5SUM.txt after download.",
    )
    return p.parse_args()


def list_gaia_files(prefix: str) -> list[str]:
    """List object keys under the given CDN77 prefix (S3-style XML)."""
    url = LIST_BASE + "/"
    resp = requests.get(url, params={"prefix": prefix, "delimiter": "/"}, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    keys: list[str] = []
    # Strip namespace if present (S3 returns default-ns XML).
    for elem in root.iter():
        if elem.tag.endswith("}Key") or elem.tag == "Key":
            if elem.text:
                keys.append(elem.text)
    return keys


def fetch_md5sums(prefix: str) -> dict[str, str]:
    """Download the _MD5SUM.txt sidecar if available and parse key→md5."""
    keys = list_gaia_files(prefix)
    md5_key = next((k for k in keys if k.endswith("_MD5SUM.txt")), None)
    if not md5_key:
        return {}
    url = f"{DOWNLOAD_BASE}/{md5_key}"
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        return {}
    mapping: dict[str, str] = {}
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        md5, name = parts[0], parts[-1]
        mapping[name] = md5
    return mapping


def download_one(key: str, dest: Path, force: bool) -> str:
    """Stream-download one shard. Returns the local path.

    Files smaller than the expected Content-Length are deleted and
    re-fetched; this protects against partial downloads left behind
    by an aborted run.
    """
    url = f"{DOWNLOAD_BASE}/{key}"
    expected_size = None
    try:
        head = requests.head(url, timeout=60, allow_redirects=True)
        if head.status_code == 200:
            expected_size = int(head.headers.get("Content-Length", 0)) or None
    except Exception:  # noqa: BLE001 — best-effort
        expected_size = None

    if dest.exists() and not force:
        if expected_size is None or dest.stat().st_size == expected_size:
            return str(dest)
        # size mismatch → partial file, delete and re-fetch
        print(f"  partial file detected ({dest.stat().st_size}/{expected_size} bytes); re-downloading", flush=True)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", expected_size or 0))
        with open(dest, "wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))
    return str(dest)


def verify_md5(path: Path, expected: str) -> bool:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected.lower()


def download_table(table_name: str, raw_dir: Path, workers: int, force: bool, verify: bool) -> list[Path]:
    """List CDN shards and parallel-download them into raw_dir."""
    prefix, stem = TABLES[table_name]
    table_dir = raw_dir / stem
    print(f"\n=== {table_name} ===", flush=True)
    print(f"  Listing prefix {prefix} via {LIST_BASE}/", flush=True)
    keys = list_gaia_files(prefix)

    md5s: dict[str, str] = {}
    if verify:
        try:
            md5s = fetch_md5sums(prefix)
            if md5s:
                print(f"  Found _MD5SUM.txt with {len(md5s)} entries", flush=True)
        except Exception as exc:  # noqa: BLE001 — best-effort verification
            print(f"  (could not fetch _MD5SUM.txt: {exc})", flush=True)

    shard_keys = [k for k in keys if k.endswith(SHARD_SUFFIXES)]
    if not shard_keys:
        raise SystemExit(
            f"[!] FAILED DOWNLOAD GATE — no shards found under prefix {prefix!r}. "
            "Check the ESA CDN path or your network connection."
        )
    print(f"  {len(shard_keys)} shard(s) to download", flush=True)

    dests = [table_dir / Path(k).name for k in shard_keys]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, k, d, force) for k, d in zip(shard_keys, dests)]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"{stem} shards", unit="file"):
            f.result()

    if md5s:
        for d in dests:
            expected = md5s.get(d.name)
            if expected is None:
                continue
            ok = verify_md5(d, expected)
            print(f"    md5 {'OK ' if ok else 'BAD'} {d.name}", flush=True)
            if not ok:
                raise SystemExit(f"[!] MD5 mismatch for {d.name}")

    return dests


def _nullify(value: bytes) -> bytes:
    """Replace literal Gaia 'null' tokens (bare or quoted) with empty bytes."""
    import re

    # Match "null" inside double quotes, or bare null delimited by comma/newline/EOL.
    # Avoid replacing null substrings inside arbitrary identifiers.
    value = re.sub(rb'"null"', b'""', value)
    value = re.sub(rb'(?<=,|\n)null(?=,|\n|$)', b'', value)
    return value


def merge_shards_to_parquet(shards: list[Path], out_path: Path, fmt: str) -> int:
    """Concatenate all shards for one table into a single parquet file."""
    import pandas as pd

    if not shards:
        raise SystemExit(f"[!] FAILED DOWNLOAD GATE — no shards to merge for {out_path.name}.")

    # Normalise to the leading-dot convention.
    fmt = fmt if fmt.startswith(".") else "." + fmt

    print(f"  Reading {len(shards)} shard(s)...", flush=True)
    frames = []
    for s in tqdm(shards, desc=f"reading {out_path.stem}", unit="file"):
        # Gaia bulk ECSV files begin with a YAML preamble and use the
        # '.ecsv' dialect; pandas.read_csv can't tokenise them as a
        # single delimiter-based table, so we use astropy's ECSV reader.
        # Gaia encodes missing values as the literal string "null"
        # (not the ECSV-standard empty token), so we pre-substitute
        # those into empty strings before handing the bytes to astropy.
        if fmt == ".csv.gz":
            from astropy.io.ascii import Ecsv

            import gzip

            with gzip.open(s, "rb") as fh:
                raw = fh.read()
            raw = _nullify(raw)
            df = Ecsv().read(raw.decode("utf-8")).to_pandas()
        elif fmt in {".fits.gz", ".fits"}:
            from astropy.table import Table

            df = Table.read(s).to_pandas()
        elif fmt in {".vot.gz", ".vot"}:
            from astropy.io import votable as avot

            df = avot.parse_single_table(str(s)).to_table().to_pandas()
        elif fmt in {".xml.gz", ".xml"}:
            from astropy.table import Table

            df = Table.read(s).to_pandas()
        else:
            raise ValueError(f"unsupported shard format: {fmt}")
        frames.append(df)

    print(f"  Concatenating...", flush=True)
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"  Wrote {len(full):,} rows × {len(full.columns)} cols → {out_path}", flush=True)
    return int(len(full))


def detect_format(filename: str) -> str:
    # Multi-suffix extensions first (e.g. .csv.gz ends with both .gz and .csv).
    for ext in (".csv.gz", ".fits.gz", ".vot.gz", ".xml.gz", ".fits", ".vot", ".xml", ".csv"):
        if filename.endswith(ext):
            return ext
    raise ValueError(f"unknown format: {filename}")


def merge_nss_tables(raw_dir: Path, out_path: Path) -> int:
    """Build data/processed/gaia_nss_master.parquet from the four NSS tables."""
    import pandas as pd

    nss_stems = [
        ("two_body", "nss_two_body.parquet"),
        ("acceleration", "nss_acceleration.parquet"),
        ("spectro", "nss_spectro.parquet"),
        ("vim", "nss_vim.parquet"),
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
        raise SystemExit(
            "[!] FAILED DOWNLOAD GATE — no NSS parquet files available to merge."
        )
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"  Wrote {len(full):,} rows × {len(full.columns)} cols → {out_path}")
    return int(len(full))


def main() -> int:
    args = parse_args()
    raw_dir = REPO_ROOT / args.raw_dir

    targets: list[str] = []
    for choice in args.only:
        if choice == "all":
            targets = list(TABLES)
            break
        if choice == "nss":
            targets += [
                "nss_two_body_orbit",
                "nss_acceleration_astro",
                "nss_non_linear_spectro",
                "nss_vim_fl",
            ]
        else:
            targets.append(choice)

    seen: set[str] = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]

    for table_name in targets:
        _, stem = TABLES[table_name]
        shards = download_table(table_name, raw_dir, args.workers, args.force, verify=not args.skip_md5)
        if not shards:
            continue
        fmt = detect_format(shards[0].name)
        merged_out = raw_dir / f"{stem}.parquet"
        merge_shards_to_parquet(shards, merged_out, fmt=fmt)

    if any(t.startswith("nss_") for t in targets):
        master_out = REPO_ROOT / "data" / "processed" / "gaia_nss_master.parquet"
        print("\n=== building gaia_nss_master.parquet ===", flush=True)
        n = merge_nss_tables(raw_dir, master_out)
        print(f"  total NSS rows: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())