"""
PharmaGuard — Phase 2: Preprocessing Pipeline Runner
======================================================
Top-level script that chains the full preprocessing pipeline:
  Load processed data → Normalize drugs → Feature engineering →
  Build sklearn pipeline → Compute class weight → Split → Save

Usage:
    python run_preprocessing.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Configure logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("PharmaGuard.Preprocessing")

# ── Project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Execute the full Phase 2 preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("PharmaGuard — Phase 2: Preprocessing Pipeline")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    config_path = PROJECT_ROOT / "config.yaml"

    # ── Load processed data ────────────────────────────────────
    processed_dir = Path(config["paths"]["processed"])
    parquet_path = processed_dir / "faers_joined.parquet"

    logger.info("Loading joined data from: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    # ── Step 2.1: Normalize drug names ─────────────────────────
    from src.preprocessing.normalize_drugs import (
        apply_drug_normalization,
        build_drug_mapping,
    )

    logger.info("--- Step 2.1: Normalizing drug names ---")
    threshold = config["preprocessing"]["drug_fuzzy_threshold"]
    mapping = build_drug_mapping(df["drugname"], threshold=threshold)
    df = apply_drug_normalization(df, mapping)

    # ── Step 2.2–2.3: Feature engineering ──────────────────────
    from src.preprocessing.feature_engineering import engineer_features

    logger.info("--- Steps 2.2-2.3: Feature engineering ---")
    df = engineer_features(df, config)

    # ── Step 2.4: Build preprocessing pipeline ─────────────────
    from src.preprocessing.build_pipeline import build_preprocessor, fit_and_save_pipeline

    logger.info("--- Step 2.4: Building sklearn pipeline ---")
    cat_features = config["preprocessing"]["categorical_features"]
    num_features = config["preprocessing"]["numerical_features"]
    target_col = config["preprocessing"]["target_column"]

    # Ensure all feature columns exist; filter to available ones
    available_cat = [c for c in cat_features if c in df.columns]
    available_num = [c for c in num_features if c in df.columns]

    if len(available_cat) < len(cat_features):
        missing = set(cat_features) - set(available_cat)
        logger.warning("Missing categorical features (skipping): %s", missing)

    preprocessor = build_preprocessor(available_cat, available_num)

    # ── Step 2.5: Compute class weight ─────────────────────────
    from src.preprocessing.compute_class_weight import compute_scale_pos_weight, save_to_config

    logger.info("--- Step 2.5: Computing scale_pos_weight ---")
    weight = compute_scale_pos_weight(df[target_col])
    save_to_config(weight, config_path)

    # ── Step 2.6: Stratified split ─────────────────────────────
    from src.preprocessing.split_data import save_splits, stratified_split

    logger.info("--- Step 2.6: Stratified splitting ---")
    split_cfg = config["splitting"]
    train_df, val_df, test_df = stratified_split(
        df,
        target_col=target_col,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_state=split_cfg["random_state"],
    )

    # Save splits
    splits_dir = Path(config["paths"]["splits"])
    save_splits(train_df, val_df, test_df, splits_dir)

    # ── Fit and save pipeline on training data ─────────────────
    logger.info("--- Fitting pipeline on training data ---")
    feature_cols = available_cat + available_num
    X_train = train_df[feature_cols]

    pipeline_path = Path(config["paths"]["model_artifacts"]) / "preprocessor.pkl"
    fit_and_save_pipeline(preprocessor, X_train, pipeline_path)

    # ── Done ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2 COMPLETE")
    logger.info("  scale_pos_weight:     %.2f", weight)
    logger.info("  Train rows:           %d", len(train_df))
    logger.info("  Val rows:             %d", len(val_df))
    logger.info("  Test rows:            %d", len(test_df))
    logger.info("  Splits saved to:      %s", splits_dir)
    logger.info("  Pipeline saved to:    %s", pipeline_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
