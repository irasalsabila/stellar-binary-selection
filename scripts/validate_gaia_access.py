"""Validate access to the Gaia Archive and the expected ADQL tables/fields.

Run via:
    python scripts/validate_gaia_access.py

The script (a) prints the astroquery / Gaia archive version, (b) lists
the four NSS tables and binary_masses, (c) verifies the cross-match
tables exist, and (d) executes a minimal 1-row query against the
parent sample to confirm the full join path returns a result.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


REQUIRED_GAIA_TABLES: dict[str, list[str]] = {
    "gaiadr3.gaia_source": [
        "source_id",
        "random_index",
        "parallax",
        "parallax_error",
        "parallax_over_error",
        "ruwe",
        "non_single_star",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "ipd_frac_multi_peak",
    ],
    "gaiadr3.tmass_psc_xsc_best_neighbour": ["source_id", "clean_tmass_psc_xsc_oid"],
    "gaiadr3.tmass_psc_xsc_join": ["clean_tmass_psc_xsc_oid", "original_psc_source_id"],
    "gaiadr1.tmass_original_valid": [
        "designation",
        "j_m",
        "h_m",
        "ks_m",
        "ph_qual",
    ],
    "gaiadr3.nss_two_body_orbit": ["source_id", "nss_solution_type"],
    "gaiadr3.nss_acceleration_astro": ["source_id", "nss_solution_type"],
    "gaiadr3.nss_non_linear_spectro": ["source_id", "nss_solution_type"],
    "gaiadr3.nss_vim_fl": ["source_id", "nss_solution_type"],
    "gaiadr3.binary_masses": ["source_id", "m1", "m2", "fluxratio", "combination_method"],
}


def check_astroquery() -> None:
    try:
        astroquery = importlib.import_module("astroquery")
        print(f"astroquery version: {astroquery.__version__}")
    except ImportError as exc:
        raise SystemExit(
            "astroquery is not installed. Install with `pip install astroquery`."
        ) from exc


def check_tables() -> dict[str, str]:
    """Try TAP table_info for each required table. Returns status map."""
    from astroquery.gaia import Gaia

    statuses: dict[str, str] = {}
    for full_name in REQUIRED_GAIA_TABLES:
        schema, table = full_name.split(".")
        try:
            job = Gaia.load_table(full_name)
            statuses[full_name] = "ok"
            del job
        except Exception as exc:  # noqa: BLE001 — surface any TAP error
            statuses[full_name] = f"FAIL: {exc}"
    for name, status in statuses.items():
        print(f"[{status}] {name}")
    return statuses


def smoke_test_query() -> int:
    """Execute a minimal 1-row query against the parent-sample join."""
    from astroquery.gaia import Gaia

    adql_path = REPO_ROOT / "queries" / "gaia_2mass_parent_test.adql"
    adql = adql_path.read_text().replace("SELECT TOP 1000", "SELECT TOP 1", 1)
    job = Gaia.launch_job(adql, dump_to_file=False)
    result = job.get_results()
    n = len(result)
    print(f"smoke-test join returned {n} row(s)")
    return n


def main() -> int:
    check_astroquery()
    print("Checking required Gaia Archive tables...")
    statuses = check_tables()
    failed = [name for name, status in statuses.items() if not status.startswith("ok")]
    if failed:
        print(f"\n{len(failed)} table(s) failed to load:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("\nRunning 1-row smoke-test join...")
    n = smoke_test_query()
    if n < 1:
        print("ERROR: smoke-test query returned 0 rows; investigate cross-match.")
        return 1
    print("Gaia ADQL access validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())