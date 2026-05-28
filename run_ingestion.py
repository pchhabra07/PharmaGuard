"""
PharmaGuard — Phase 1: Data Ingestion Runner
===============================================
Top-level script that chains the full ingestion pipeline:
  Download → Extract → Load Tables → Join → Validate → Save → EDA

Usage:
    python run_ingestion.py
"""

import logging
import sys
from pathlib import Path

import yaml

# ── Configure logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("PharmaGuard.Ingestion")

# ── Project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Execute the full Phase 1 ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("PharmaGuard — Phase 1: Data Ingestion Pipeline")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    logger.info("Config loaded. Quarters to process: %s", config["faers"]["quarters"])

    # ── Step 1.2: Download FAERS ZIP(s) ────────────────────────
    from src.ingestion.download import download_all_quarters

    logger.info("--- Step 1.2: Downloading FAERS data ---")
    zip_paths = download_all_quarters(config)

    # ── Step 1.2: Extract ZIP(s) ───────────────────────────────
    from src.ingestion.extract import extract_zip, load_all_tables

    logger.info("--- Step 1.2: Extracting ZIP files ---")
    raw_dir = Path(config["paths"]["raw"])
    extracted_dirs = []
    for zp in zip_paths:
        quarter_name = zp.stem  # e.g. faers_ascii_2026q1
        extract_dest = raw_dir / quarter_name
        extracted_dir = extract_zip(zp, extract_dest)
        extracted_dirs.append(extracted_dir)

    # ── Step 1.2: Load tables ──────────────────────────────────
    logger.info("--- Step 1.2: Loading FAERS tables ---")
    table_names = config["faers"]["tables"]

    # Process the first quarter (extend for multi-quarter later)
    tables = load_all_tables(extracted_dirs[0], table_names)

    # ── Step 1.3: Join tables ──────────────────────────────────
    from src.ingestion.join import join_tables, save_joined

    logger.info("--- Step 1.3: Joining tables ---")
    serious_codes = config["preprocessing"]["serious_outcome_codes"]
    joined_df = join_tables(tables, serious_codes)

    # ── Step 1.4: Validate schema ──────────────────────────────
    from src.ingestion.validate import validate_schema

    logger.info("--- Step 1.4: Validating schema ---")
    warnings = validate_schema(joined_df)
    if warnings:
        logger.warning("Validation produced %d warning(s) — review above.", len(warnings))

    # ── Save joined data ───────────────────────────────────────
    processed_dir = Path(config["paths"]["processed"])
    save_joined(joined_df, processed_dir)

    # ── Step 1.5: Exploratory Data Analysis ────────────────────
    from src.ingestion.eda import run_eda

    logger.info("--- Step 1.5: Running EDA ---")
    eda_dir = Path(config["paths"]["eda"])
    run_eda(joined_df, eda_dir)

    # ── Done ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1 COMPLETE")
    logger.info("  Joined data saved to: %s", processed_dir)
    logger.info("  EDA plots saved to:   %s", eda_dir)
    logger.info("  Total rows:           %d", len(joined_df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
