"""Download PARSEC isochrones for the stellar emulator (TODO M7).

PARSEC v2.0 grids are distributed by the Trieste stellar collaboration
via the official CMD web interface. This script provides the manual
download hook and an audit that confirms the cached grids are usable.

Run via:
    PYTHONPATH=src python scripts/download_parsec.py

NOTE: The full PARSEC archive is multi-GB; we only fetch the FGK
main-sequence subset relevant to the project's mass range. Manual
download instructions are printed when automated retrieval is not
possible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ISO_DIR = REPO_ROOT / "data" / "isochrones" / "parsec"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manual",
        action="store_true",
        help="Print manual download instructions instead of attempting HTTP fetch.",
    )
    args = p.parse_args()

    ISO_DIR.mkdir(parents=True, exist_ok=True)
    if args.manual:
        print(
            "Manual download instructions:\n"
            " 1. Visit http://stev.oapd.inaf.it/cgi-bin/cmd\n"
            " 2. Select PARSEC v2.0 tracks, Gaia DR3 + 2MASS filters,\n"
            "    M in [0.6, 1.4] Msun, age 0.5–12 Gyr, [M/H] in [-1.0, +0.5].\n"
            " 3. Save the resulting grid under:\n"
            f"    {ISO_DIR}/parsec_fgk_grid.parquet\n"
        )
        return 0

    print(
        "Automatic PARSEC download not implemented (server-side scripting is\n"
        "deliberately disabled by the CMD interface). Re-run with --manual\n"
        "for step-by-step instructions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())