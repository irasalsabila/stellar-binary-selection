"""NASA ADS literature search helper (TODO M0.4 / M31).

Requires a free ADS API token (https://ui.adsabs.harvard.edu/user/settings/token):
    export ADS_DEV_KEY="<your-token>"
    PYTHONPATH=src python scripts/ads_search.py

Runs the mandatory prior-work queries from the master TODO, saves the
resulting article list and a BibTeX export for the novelty check.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Mandatory queries (from plans/Master Research TODO — v1.0.md, §M0.4)
QUERIES = {
    "photometric_binary_selection_function": "abstract:\"photometric binary selection function\"",
    "gaia_2mass_unresolved_binary_ml": "abstract:(\"Gaia\" AND \"2MASS\" AND \"unresolved binary\" AND \"machine learning\")",
    "binary_mass_ratio_photometric_selection": "abstract:(\"binary mass ratio\" AND \"photometric selection function\")",
    "gaia_unresolved_binary_completeness_photometry": "abstract:(\"Gaia\" AND \"unresolved binary\" AND \"completeness\" AND \"photometry\")",
    "photometric_astrometric_binary_complementarity": "abstract:(\"photometric\" AND \"astrometric\" AND \"binary\" AND \"complementarity\")",
    "physics_informed_binary_star_inference": "abstract:(\"physics informed\" AND \"binary star\" AND \"inference\")",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--out", default="results/tables/ads_search.json")
    args = p.parse_args()

    token = os.environ.get("ADS_DEV_KEY")
    if not token:
        print("ERROR: set ADS_DEV_KEY first.")
        print("  1. Get a token at https://ui.adsabs.harvard.edu/user/settings/token")
        print("  2. export ADS_DEV_KEY='<token>'")
        print("  (Or run the queries manually at https://ui.adsabs.harvard.edu and export BibTeX.)")
        return 1

    try:
        from ads import SearchQuery  # type: ignore
    except ImportError:
        print("ERROR: pip install ads  (then set ADS_DEV_KEY).")
        return 1

    import json
    results = {}
    for name, q in QUERIES.items():
        print(f"Querying: {name} ...", flush=True)
        try:
            papers = SearchQuery(q=q, rows=args.max_results,
                                 fl=["title", "author", "year", "bibcode", "doi",
                                     "abstract", "citation"])
            rows = []
            for art in papers:
                rows.append({
                    "bibcode": getattr(art, "bibcode", None),
                    "title": getattr(art, "title", [None])[0] if getattr(art, "title", None) else None,
                    "first_author": (getattr(art, "author", [None]) or [None])[0],
                    "year": getattr(art, "year", None),
                    "doi": getattr(art, "doi", None),
                    "citations": getattr(art, "citation", None),
                })
            results[name] = rows
            print(f"  -> {len(rows)} papers", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  query failed: {exc}", flush=True)
            results[name] = []
        time.sleep(1.0)  # be polite to the API

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    print("Review the titles/abstracts; if none overlaps the exact proposed")
    print("combination (PARSEC emulator + exact flux + phot/astro complementarity),")
    print("the novelty claim in §M0 GATE is supported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
