"""
convert_rds_to_json.py — Convert legacy R .rds files to JSON.

Reads RDS files from archive/legacy/fantasy_banter_data/{year}/ using pyreadr,
converts each DataFrame to JSON, and writes to data/raw/supercoach/{year}/.

Usage
-----
  py convert_rds_to_json.py --year 2020
  py convert_rds_to_json.py --year 2021
  py convert_rds_to_json.py         # converts all available years

Install dependency first:
  pip install pyreadr

This is a one-off utility for importing historical R data into the project.
Not part of the weekly pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_DIR = PROJECT_ROOT / "archive" / "legacy" / "fantasy_banter_data"
OUTPUT_BASE = PROJECT_ROOT / "data" / "raw" / "supercoach"

YEARS = [2020, 2021]


def convert_year(year: int) -> int:
    """Convert all .rds files for a given year. Returns count of files converted."""
    try:
        import pyreadr
    except ImportError:
        log.error("pyreadr not installed. Run: pip install pyreadr")
        return 0

    src_dir = LEGACY_DIR / str(year)
    dst_dir = OUTPUT_BASE / str(year)
    dst_dir.mkdir(parents=True, exist_ok=True)

    rds_files = sorted(src_dir.glob("*.rds")) + sorted(src_dir.glob("*.RDS"))
    if not rds_files:
        log.warning(f"No .rds files found in {src_dir}")
        return 0

    converted = 0
    for rds_path in rds_files:
        out_path = dst_dir / (rds_path.stem + ".json")
        try:
            result = pyreadr.read_r(str(rds_path))
            # pyreadr returns an OrderedDict of DataFrames; take the first (or only) one
            data = next(iter(result.values()))
            data.to_json(out_path, orient="records", indent=2)
            log.info(f"  {rds_path.name} → {out_path.name} ({len(data)} rows)")
            converted += 1
        except Exception as exc:
            log.warning(f"  Failed to convert {rds_path.name}: {exc}")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert legacy R .rds files to JSON.")
    parser.add_argument("--year", type=int, choices=YEARS,
                        help=f"Year to convert ({', '.join(str(y) for y in YEARS)}). Omit to convert all.")
    args = parser.parse_args()

    years = [args.year] if args.year else YEARS
    total = 0
    for year in years:
        log.info(f"Converting {year}...")
        n = convert_year(year)
        log.info(f"  {n} files converted for {year}.")
        total += n

    log.info(f"Done. {total} total files converted → {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
