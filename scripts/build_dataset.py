"""Build derived observational quantities from the raw Gaia+2MASS catalogue.

Run via:
    python scripts/build_dataset.py --stage audit
    python scripts/build_dataset.py --stage derived
    python scripts/build_dataset.py --stage cmd
    python scripts/build_dataset.py --stage fgk_cut

Implements TODO M4.1–M4.4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc="rows")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.utils import hash_dataframe, write_run_metadata  # noqa: E402


RAW_PATH = REPO_ROOT / "data" / "raw" / "gaia_dr3_2mass_parent.parquet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


DERIVED_COLUMNS = [
    "bp_rp",
    "g_j",
    "g_ks",
    "j_h",
    "h_ks",
    "j_ks",
    "distance_pc_prelim",
    "m_g_prelim",
]


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise SystemExit(
            f"Raw catalogue not found at {RAW_PATH}. Run scripts/download_gaia.py first."
        )
    return pd.read_parquet(RAW_PATH)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute colour indices and exploratory distances / M_G.

    These preliminary distances use the naive 1000/parallax estimator and
    are only valid for very high-S/N sources. The full PI-NN pipeline
    uses a parallax likelihood rather than 1000/parallax directly.
    """
    print(f"  computing colour indices and M_G for {len(df):,} rows...", flush=True)
    out = df.copy()
    out["bp_rp"] = out["phot_bp_mean_mag"] - out["phot_rp_mean_mag"]
    out["g_j"] = out["phot_g_mean_mag"] - out["j_m"]
    out["g_ks"] = out["phot_g_mean_mag"] - out["ks_m"]
    out["j_h"] = out["j_m"] - out["h_m"]
    out["h_ks"] = out["h_m"] - out["ks_m"]
    out["j_ks"] = out["j_m"] - out["ks_m"]

    # Naive distance and absolute magnitude for exploratory work only.
    parallax = out["parallax"].clip(lower=0.1)  # avoid div-by-zero
    out["distance_pc_prelim"] = 1000.0 / parallax
    out["m_g_prelim"] = (
        out["phot_g_mean_mag"]
        - 5.0 * np.log10(out["distance_pc_prelim"])
        + 5.0
    )
    return out


def audit(df: pd.DataFrame) -> dict[str, int]:
    """Run the M4.1 basic validation checks."""
    report: dict[str, int] = {
        "n_rows": int(len(df)),
        "n_unique_source_id": int(df["source_id"].nunique()),
        "n_duplicated_source_id": int(df["source_id"].duplicated().sum()),
        "n_missing_ruwe": int(df["ruwe"].isna().sum()),
        "n_missing_g_mag": int(df["phot_g_mean_mag"].isna().sum()),
        "n_missing_bp_mag": int(df["phot_bp_mean_mag"].isna().sum()),
        "n_missing_rp_mag": int(df["phot_rp_mean_mag"].isna().sum()),
        "n_missing_j_m": int(df["j_m"].isna().sum()),
        "n_missing_h_m": int(df["h_m"].isna().sum()),
        "n_missing_ks_m": int(df["ks_m"].isna().sum()),
        "n_inf_parallax": int(np.isinf(df["parallax"].to_numpy()).sum()),
        "n_negative_parallax": int((df["parallax"] < 0).sum()),
        "n_negative_flux": int(
            (df[["phot_g_mean_flux", "phot_bp_mean_flux", "phot_rp_mean_flux"]] < 0)
            .any(axis=1)
            .sum()
        ),
        "n_low_ruwe": int((df["ruwe"] < 1.4).sum()),
        "n_high_ruwe": int((df["ruwe"] >= 1.4).sum()),
        "n_non_single_star": int((df["non_single_star"] > 0).sum()),
    }
    return report


def save_derived(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "gaia_dr3_2mass_derived.parquet"
    df.to_parquet(out, index=False)
    return out


def plot_cmd(df: pd.DataFrame, output: Path) -> None:
    """Produce Figure 1 — observed Gaia CMD with derived FGK selection."""
    import matplotlib.pyplot as plt

    print(f"  rendering CMD figure ({len(df):,} points)...", flush=True)
    fig, ax = plt.subplots(figsize=(7, 9))
    sel = np.isfinite(df["bp_rp"]) & np.isfinite(df["m_g_prelim"])
    n_plot = int(sel.sum())
    # Subsample to keep figure size reasonable on very large catalogues.
    if n_plot > 200_000:
        idx = np.random.default_rng(0).choice(np.where(sel)[0], 200_000, replace=False)
        bp = df["bp_rp"].to_numpy()[idx]
        mg = df["m_g_prelim"].to_numpy()[idx]
    else:
        bp = df.loc[sel, "bp_rp"].to_numpy()
        mg = df.loc[sel, "m_g_prelim"].to_numpy()
    ax.scatter(bp, mg, s=1, c="k", alpha=0.2)
    ax.set_xlabel("BP - RP [mag]")
    ax.set_ylabel("M_G (preliminary) [mag]")
    ax.invert_yaxis()
    ax.set_title(f"Gaia DR3 + 2MASS preliminary CMD ({n_plot:,} sources)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def fgk_main_sequence_envelope(df: pd.DataFrame) -> tuple[float, float]:
    """A provisional FGK cut: 0.5 < BP-RP < 1.6 and M_G between 3 and 8.

    Boundaries are intentionally generous in M4; they will be tightened
    against PARSEC isochrones in M7.
    """
    bp_rp_lo, bp_rp_hi = 0.5, 1.6
    mg_lo, mg_hi = 3.0, 8.0
    return (bp_rp_lo, bp_rp_hi), (mg_lo, mg_hi)


def apply_fgk_cut(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    (bp_lo, bp_hi), (mg_lo, mg_hi) = fgk_main_sequence_envelope(df)
    sel = (
        df["bp_rp"].between(bp_lo, bp_hi)
        & df["m_g_prelim"].between(mg_lo, mg_hi)
    )
    cut = df.loc[sel].copy()
    report = {
        "n_input": int(len(df)),
        "n_after_fgk": int(len(cut)),
        "bp_rp_range": [float(bp_lo), float(bp_hi)],
        "m_g_range": [float(mg_lo), float(mg_hi)],
    }
    return cut, report


# ---------------------------------------------------------------------------
# M5 — real-data samples (A = parent, B = apparently-single, C = NSS positives)
# ---------------------------------------------------------------------------

NSS_MASTER_PATH = (
    REPO_ROOT / "data" / "processed" / "gaia_nss_master.parquet"
)


def join_nss(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Annotate the parent sample with NSS membership (Sample C sources).

    A source_id may appear multiple times in the NSS master (multiple
    orbital/spectro solutions across tables), so we aggregate per source
    and record how many distinct tables flagged it.
    """
    if not NSS_MASTER_PATH.exists():
        raise SystemExit(
            f"NSS master not found at {NSS_MASTER_PATH}. Run scripts/process_gaia_bulk.py first."
        )
    print(f"  loading {NSS_MASTER_PATH.name}...", flush=True)
    nss = pd.read_parquet(NSS_MASTER_PATH)

    if "source_id" not in nss.columns:
        raise SystemExit(f"NSS master is missing a 'source_id' column; found {nss.columns[:8]}")

    agg = (
        nss.groupby("source_id")
        .agg(
            nss_n_solutions=("source_id", "size"),
            nss_tables=(
                "nss_table_origin",
                lambda s: ",".join(sorted(set(s))) if "nss_table_origin" in nss.columns else "",
            ),
        )
        .reset_index()
    )
    if "nss_solution_type" in nss.columns:
        types = (
            nss.groupby("source_id")["nss_solution_type"]
            .apply(lambda s: ",".join(sorted({str(x) for x in s if pd.notna(x)})))
            .rename("nss_solution_types")
        )
        agg = agg.merge(types, on="source_id", how="left")

    out = df.merge(agg, on="source_id", how="left")
    out["nss_n_solutions"] = out["nss_n_solutions"].fillna(0).astype(int)
    out["is_nss"] = out["nss_n_solutions"] > 0
    for c in ("nss_tables", "nss_solution_types"):
        if c in out.columns:
            out[c] = out[c].fillna("")

    report = {
        "n_parent": int(len(df)),
        "n_nss_distinct_sources": int(len(agg)),
        "n_parent_matched_to_nss": int(out["is_nss"].sum()),
        "n_nss_solutions_total": int(len(nss)),
    }
    for t in ("two_body", "acceleration", "spectro", "vim"):
        if "nss_tables" in out.columns:
            report[f"n_parent_{t}"] = int(out["nss_tables"].str.contains(t).sum())
    return out, report


def build_sample_b(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sample B — 'apparently single' conservative training set.

    Never call this 'confirmed single'. Criteria per PRD §14:
      - non_single_star == 0
      - RUWE below a conservative empirical percentile cut
      - ipd_frac_multi_peak == 0
      - no blending/contamination flags
      - high parallax S/N (already guaranteed by the parent query)
    """
    print("  determining empirical RUWE criterion...", flush=True)
    ruwe = pd.to_numeric(df["ruwe"], errors="coerce")
    # Conservative: 95th percentile of the low-RUWE mode, i.e. the bulk
    # of well-behaved single-star solutions.
    ruwe_cut = float(np.nanpercentile(ruwe[ruwe < 1.4], 95))
    print(f"    empirical RUWE criterion = {ruwe_cut:.3f}", flush=True)

    sel = (
        (pd.to_numeric(df["non_single_star"], errors="coerce") == 0)
        & (ruwe <= ruwe_cut)
        & (pd.to_numeric(df.get("ipd_frac_multi_peak", 0), errors="coerce") == 0)
    )
    # blending / contamination: no contaminated or blended transits
    for c in (
        "phot_bp_n_contaminated_transits",
        "phot_bp_n_blended_transits",
        "phot_rp_n_contaminated_transits",
        "phot_rp_n_blended_transits",
    ):
        if c in df.columns:
            sel &= (pd.to_numeric(df[c], errors="coerce").fillna(0) == 0)

    sample_b = df.loc[sel].copy()
    report = {
        "n_input": int(len(df)),
        "n_sample_b": int(len(sample_b)),
        "ruwe_cut": ruwe_cut,
        "sample_b_fraction": float(len(sample_b) / max(len(df), 1)),
    }
    return sample_b, report


def build_sample_c(df: pd.DataFrame, nss_master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sample C — NSS-positive validation set, one row per (source × solution)."""
    if "source_id" not in nss_master.columns:
        raise SystemExit("NSS master is missing 'source_id'.")
    parent_ids = set(df["source_id"])
    sel = nss_master["source_id"].isin(parent_ids)
    sample_c = nss_master.loc[sel].copy()
    report = {
        "n_nss_solutions": int(len(nss_master)),
        "n_in_parent": int(sel.sum()),
        "n_sample_c_rows": int(len(sample_c)),
        "n_distinct_sources": int(sample_c["source_id"].nunique()),
    }
    if "nss_table_origin" in sample_c.columns:
        report["by_table"] = sample_c["nss_table_origin"].value_counts().to_dict()
    return sample_c, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["audit", "derived", "cmd", "fgk_cut", "samples", "all"],
        default="all",
    )
    parser.add_argument(
        "--metadata-dir",
        default="results/runs/build_dataset",
    )
    args = parser.parse_args()

    print("Loading raw parquet...", flush=True)
    df = load_raw()
    print(f"  loaded {len(df):,} rows", flush=True)
    metrics: dict = {"raw_sha256": hash_dataframe(df), "n_rows": len(df)}

    if args.stage in {"audit", "all"}:
        print("Stage: audit...", flush=True)
        metrics["audit"] = audit(df)

    if args.stage in {"derived", "all"}:
        print("Stage: derived quantities (colours, M_G)...", flush=True)
        df = add_derived(df)
        metrics["n_rows_after_derived"] = len(df)
        out = save_derived(df)
        metrics["derived_parquet"] = str(out.relative_to(REPO_ROOT))
        print(f"  wrote {out.relative_to(REPO_ROOT)}", flush=True)

    if args.stage in {"cmd", "all"}:
        print("Stage: plotting CMD figure...", flush=True)
        df = add_derived(df) if "bp_rp" not in df.columns else df
        fig_path = REPO_ROOT / "results" / "figures" / "fig1_cmd.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_cmd(df, fig_path)
        metrics["cmd_figure"] = str(fig_path.relative_to(REPO_ROOT))
        print(f"  wrote {fig_path.relative_to(REPO_ROOT)}", flush=True)

    if args.stage in {"fgk_cut", "all"}:
        print("Stage: FGK main-sequence cut...", flush=True)
        df = add_derived(df) if "bp_rp" not in df.columns else df
        cut, report = apply_fgk_cut(df)
        cut_path = PROCESSED_DIR / "gaia_dr3_2mass_fgk.parquet"
        cut_path.parent.mkdir(parents=True, exist_ok=True)
        cut.to_parquet(cut_path, index=False)
        metrics["fgk_cut"] = report
        metrics["fgk_parquet"] = str(cut_path.relative_to(REPO_ROOT))
        print(f"  {report['n_after_fgk']:,} / {report['n_input']:,} rows passed", flush=True)

    if args.stage in {"samples", "all"}:
        df = add_derived(df) if "bp_rp" not in df.columns else df

        # --- Sample A: the full parent sample (no RUWE / NSS filtering) ---
        print("Stage: Sample A (parent)...", flush=True)
        sample_a_path = PROCESSED_DIR / "sample_a_parent.parquet"
        df.to_parquet(sample_a_path, index=False)
        metrics["sample_a"] = {"n_rows": int(len(df)), "path": str(sample_a_path.relative_to(REPO_ROOT))}
        print(f"  {len(df):,} rows", flush=True)

        # --- Join NSS onto the parent so we know which sources are NSS ---
        print("Stage: NSS join...", flush=True)
        df_nss, nss_report = join_nss(df)
        df = df_nss
        annotated_path = PROCESSED_DIR / "gaia_dr3_2mass_derived_nss.parquet"
        df.to_parquet(annotated_path, index=False)
        metrics["nss_join"] = nss_report
        print(f"  {nss_report['n_parent_matched_to_nss']:,} parent sources are NSS-positive", flush=True)

        # --- Sample B: apparently single (deliberately conservative) ---
        print("Stage: Sample B (apparently single)...", flush=True)
        sample_b, b_report = build_sample_b(df)
        sample_b_path = PROCESSED_DIR / "sample_b_apparently_single.parquet"
        sample_b.to_parquet(sample_b_path, index=False)
        metrics["sample_b"] = b_report
        print(f"  {b_report['n_sample_b']:,} rows (RUWE <= {b_report['ruwe_cut']:.3f})", flush=True)

        # --- Sample C: NSS-positive validation set ---
        print("Stage: Sample C (NSS positives)...", flush=True)
        nss_master = pd.read_parquet(NSS_MASTER_PATH)
        sample_c, c_report = build_sample_c(df, nss_master)
        sample_c_path = PROCESSED_DIR / "sample_c_nss.parquet"
        sample_c.to_parquet(sample_c_path, index=False)
        metrics["sample_c"] = c_report
        print(f"  {c_report['n_sample_c_rows']:,} rows across {c_report['n_distinct_sources']:,} distinct sources", flush=True)

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={"stage": args.stage, "raw_path": str(RAW_PATH.relative_to(REPO_ROOT))},
        metrics=metrics,
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())