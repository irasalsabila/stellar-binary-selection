"""Merge the four PARSEC isochrone tables (Gaia EDR3 + 2MASS, low + high [M/H])
into the canonical `parsec_fgk_grid.parquet` consumed by
`stellar_binary_selection.physics.isochrones.ParsecGrid`.

Expected files (CMD 3.9, PARSEC v1.2S, OBC bolometric corrections):
    data/isochrones/parsec/raw/parsec_gaia_edr3_mh_low.dat.txt
    data/isochrones/parsec/raw/parsec_gaia_edr3_mh_high.dat.txt
    data/isochrones/parsec/raw/parsec_2mass_mh_low.dat.txt
    data/isochrones/parsec/raw/parsec_2mass_mh_high.dat.txt

Each file has columns:
    Zini, MH, logAge, Mini, int_IMF, Mass, logL, logTe, logg, label, mbolmag,
    Gmag, G_BPmag, G_RPmag          (Gaia file)
    Jmag, Hmag, Ksmag              (2MASS file)

The merger joins the two photometric systems on the common physical
coordinates (MH, logAge, Mini) and emits a single parquet with the
combined absolute magnitudes in the standard column names expected by
the pipeline:

    mass   g  bp  rp  j  h  ks    (absolute magnitudes in AB / Vega)
    feh, log_age                       (auxiliary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


RAW_DIR = REPO_ROOT / "data" / "isochrones" / "parsec" / "raw"
OUT_PATH = REPO_ROOT / "data" / "isochrones" / "parsec" / "parsec_fgk_grid.parquet"

GAIA_FILES = [
    RAW_DIR / "parsec_gaia_edr3_mh_low.dat.txt",
    RAW_DIR / "parsec_gaia_edr3_mh_high.dat.txt",
]
TWOMASS_FILES = [
    RAW_DIR / "parsec_2mass_mh_low.dat.txt",
    RAW_DIR / "parsec_2mass_mh_high.dat.txt",
]


def parse_cmd_isochrone(path: Path) -> pd.DataFrame:
    """Read a CMD 3.9 isochrone table, returning a clean DataFrame.

    The file has a leading block of '#'-commented metadata and a column
    header line. We strip the comments, then read the data with the
    column names embedded in the last commented header line.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"missing isochrone file: {path}. "
            "Run scripts/download_parsec.py --manual to obtain it from the CMD interface."
        )
    with open(path) as f:
        text = f.read()
    lines = text.splitlines()
    # Last commented line before data is the column header.
    header_line = None
    for line in lines:
        s = line.strip()
        if s.startswith("#") and not s.startswith("##") and not s.startswith("#!"):
            # Skip global metadata lines but keep the column header that
            # starts with a column name (lowercase letter).
            tokens = s.lstrip("#").split()
            if tokens and (
                tokens[0].lower() in ("zini", "z_ini")
                or "logage" in s.lower()
                and "mag" in s.lower()
            ):
                header_line = s.lstrip("#").strip()
        elif s and not s.startswith("#"):
            break
    if header_line is None:
        # Some files have the header twice — use the last '#'-prefixed line.
        for line in reversed(lines):
            s = line.strip()
            if s.startswith("#") and "logAge" in s:
                header_line = s.lstrip("#").strip()
                break
    if header_line is None:
        raise ValueError(f"no column header found in {path}")

    cols = header_line.split()
    # Skip the comment block and any data lines.
    data_lines = [
        ln for ln in lines
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    from io import StringIO
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=cols,
        engine="python",
    )
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT_PATH), help="Output parquet path.")
    p.add_argument(
        "--mass-min", type=float, default=0.6,
        help="Lower mass cut (Msun) applied to the final grid.",
    )
    p.add_argument(
        "--mass-max", type=float, default=1.4,
        help="Upper mass cut (Msun) applied to the final grid.",
    )
    args = p.parse_args()

    print("Loading Gaia EDR3 isochrones (low + high metallicity halves)...", flush=True)
    gaia_low = parse_cmd_isochrone(RAW_DIR / "parsec_gaia_edr3_mh_low.dat.txt")
    gaia_high = parse_cmd_isochrone(RAW_DIR / "parsec_gaia_edr3_mh_high.dat.txt")
    print(f"  Gaia low  rows: {len(gaia_low):,}  MH ∈ [{gaia_low['MH'].min()}, {gaia_low['MH'].max()}]")
    print(f"  Gaia high rows: {len(gaia_high):,}  MH ∈ [{gaia_high['MH'].min()}, {gaia_high['MH'].max()}]")
    gaia = pd.concat([gaia_low, gaia_high], ignore_index=True)
    print(f"  Gaia combined rows: {len(gaia):,}", flush=True)

    print("Loading 2MASS isochrones (concatenated to cover -0.2..+0.5 in one file)...", flush=True)
    tm_low = parse_cmd_isochrone(RAW_DIR / "parsec_2mass_mh_low.dat.txt")
    tm_high = parse_cmd_isochrone(RAW_DIR / "parsec_2mass_mh_high.dat.txt")
    print(f"  2MASS low  rows: {len(tm_low):,}  MH ∈ [{tm_low['MH'].min()}, {tm_low['MH'].max()}]")
    print(f"  2MASS high rows: {len(tm_high):,}  MH ∈ [{tm_high['MH'].min()}, {tm_high['MH'].max()}]")
    twomass = pd.concat([tm_low, tm_high], ignore_index=True)
    print(f"  2MASS combined rows: {len(twomass):,}", flush=True)

    # Keep only main-sequence / subgiant stages: drop the thermally-pulsing
    # AGB and beyond by filtering on the evolutionary label. label == 0 is
    # pre-MS, label == 1 is MS, label == 2 is subgiant, label >= 3 are
    # later phases. We keep labels 0–2 to include the MS turnoff and
    # subgiant branch (for the M1 upper end).
    gaia = gaia[gaia["label"] <= 2].copy()
    twomass = twomass[twomass["label"] <= 2].copy()
    print(f"  after evolutionary cut: Gaia={len(gaia):,}, 2MASS={len(twomass):,}", flush=True)

    # Mass cut (FGK regime).
    gaia = gaia[gaia["Mass"].between(args.mass_min, args.mass_max)].copy()
    twomass = twomass[twomass["Mass"].between(args.mass_min, args.mass_max)].copy()
    print(f"  after mass cut [{args.mass_min}, {args.mass_max}]: Gaia={len(gaia):,}, 2MASS={len(twomass):,}", flush=True)

    # Join Gaia (spine) with 2MASS via LEFT JOIN, keyed on the common
    # physical coordinates. Duplicates (MH, logAge, Mass) can appear
    # because the isochrone mass grid is non-uniform; collapse each side
    # by mean of the photometric columns before merging. The 2MASS file
    # we have only covers MH >= -0.2, so for the low-metallicity Gaia
    # points (MH < -0.2) the J/H/Ks columns will be NaN. That's flagged
    # in the diagnostic and propagated downstream.
    join_keys = ["MH", "logAge", "Mass"]

    keep_gaia = ["MH", "logAge", "Mass", "Gmag", "G_BPmag", "G_RPmag", "mbolmag",
                 "Zini", "Mini", "logL", "logTe", "logg"]
    keep_gaia = [c for c in keep_gaia if c in gaia.columns]
    agg_gaia = gaia.groupby(join_keys, as_index=False)[keep_gaia].mean()
    print(f"  Gaia unique (MH, logAge, Mass) tuples: {len(agg_gaia):,}", flush=True)

    agg_twomass = (
        twomass.groupby(join_keys, as_index=False)[["Jmag", "Hmag", "Ksmag"]]
        .mean()
    )
    print(f"  2MASS unique (MH, logAge, Mass) tuples: {len(agg_twomass):,}", flush=True)

    # LEFT JOIN: keep all Gaia points; fill missing 2MASS with NaN.
    merged = agg_gaia.merge(agg_twomass, on=join_keys, how="left")
    print(f"  merged rows: {len(merged):,}", flush=True)
    n_no_2mass = int(merged["Jmag"].isna().sum())
    print(f"  rows with no 2MASS coverage: {n_no_2mass:,}", flush=True)
    if len(merged) < 100:
        print("  WARNING: very few merged rows — check that the Gaia and 2MASS grids use the same physical points.", flush=True)

    # Rename to the pipeline's expected column names. The downstream
    # `ParsecGrid` mapper accepts both `g_mag / gaia_g` etc.; we provide
    # the canonical set here.
    out = pd.DataFrame({
        "MH": merged["MH"],
        "Zini": merged["Zini"],
        "logAge": merged["logAge"],
        "Mini": merged["Mini"],
        "Mass": merged["Mass"],
        "logL": merged["logL"],
        "logTe": merged["logTe"],
        "logg": merged["logg"],
        "mbolmag": merged["mbolmag"],
        "g_mag": merged["Gmag"],
        "bp_mag": merged["G_BPmag"],
        "rp_mag": merged["G_RPmag"],
        "j_mag": merged["Jmag"],
        "h_mag": merged["Hmag"],
        "ks_mag": merged["Ksmag"],
    })
    out = out.dropna().reset_index(drop=True)
    print(f"  rows with all 6 bands: {len(out):,}", flush=True)

    # Diagnostic: range of mass / logAge / MH.
    print(
        f"  mass range: [{out['Mass'].min():.3f}, {out['Mass'].max():.3f}] Msun",
        flush=True,
    )
    print(
        f"  logAge range: [{out['logAge'].min():.3f}, {out['logAge'].max():.3f}]",
        flush=True,
    )
    print(
        f"  MH range: [{out['MH'].min():.3f}, {out['MH'].max():.3f}]",
        flush=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        f"\nWrote {out_path.relative_to(REPO_ROOT)}  ({len(out):,} rows, {out.shape[1]} cols)",
        flush=True,
    )

    # Also write a Gaia-only grid covering the full [M/H] = -1.0 to +0.5
    # range, for the metal-poor end where no 2MASS coverage is available.
    full = pd.DataFrame({
        "MH": merged["MH"],
        "Zini": merged["Zini"],
        "logAge": merged["logAge"],
        "Mini": merged["Mini"],
        "Mass": merged["Mass"],
        "logL": merged["logL"],
        "logTe": merged["logTe"],
        "logg": merged["logg"],
        "mbolmag": merged["mbolmag"],
        "g_mag": merged["Gmag"],
        "bp_mag": merged["G_BPmag"],
        "rp_mag": merged["G_RPmag"],
        "j_mag": merged["Jmag"],  # NaN for MH < -0.2
        "h_mag": merged["Hmag"],
        "ks_mag": merged["Ksmag"],
    }).reset_index(drop=True)
    full_path = out_path.with_name("parsec_fgk_grid_gaia_only.parquet")
    full.to_parquet(full_path, index=False)
    print(
        f"Wrote {full_path.relative_to(REPO_ROOT)}  ({len(full):,} rows, NaN J/H/Ks for MH<{-0.2})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())