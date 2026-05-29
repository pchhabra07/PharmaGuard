"""
PharmaGuard — Phase 3: Model Training & Evaluation Runner
============================================================
Top-level script that chains the full training pipeline:
  Load config -> Load preprocessor -> Load splits -> Transform features ->
  Build XGBoost -> Train with early stopping -> Tune threshold ->
  Evaluate all splits -> Generate plots -> Save model -> Write model card ->
  Log to MLflow -> Save metrics history

Usage:
    python run_training.py
"""

import logging
import sys
from pathlib import Path

import yaml

# -- Configure logging --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("PharmaGuard.Training")

# -- Project root -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Execute the full Phase 3 training and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("PharmaGuard — Phase 3: Model Training & Evaluation")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    config_path = PROJECT_ROOT / "config.yaml"

    # -- Step 3.0: Determine Model Version --------------------------
    from src.tracking.metrics_store import load_metrics_history
    metrics_path = PROJECT_ROOT / "metrics" / "metrics_history.json"
    history = load_metrics_history(metrics_path)
    current_version_num = len(history) + 1
    model_version = f"v{current_version_num}"
    logger.info("--- Model Version: %s ---", model_version)

    # -- Step 3.0.1: Initialize experiment tracking -------------------
    from src.tracking.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(config)
    tracker.start_run(run_name=f"training-run-{model_version}")
    tracker.log_params(config.get("training", {}), prefix="training")
    tracker.log_params(
        {"quarters": str(config.get("faers", {}).get("quarters", []))},
        prefix="data",
    )

    # -- Step 3.1: Load the fitted preprocessor ---------------------
    from src.training.utils import (
        get_feature_names,
        load_preprocessor,
        load_split,
        transform_features,
    )

    logger.info("--- Step 3.1: Loading preprocessor ---")
    
    # We load preprocessor from the current API version, but wait, the preprocessor isn't trained here. 
    # Actually, the preprocessor should just be loaded from `model_artifacts/preprocessor.pkl` if it hasn't been versioned yet.
    # However, since preprocessing is separate, let's load it from the root of model_artifacts as it was, and then save it to the versioned directory!
    
    preprocessor_src_path = Path(config["paths"]["model_artifacts"]) / "preprocessor.pkl"
    if not preprocessor_src_path.exists():
        logger.error("Preprocessor not found at %s. Please run run_preprocessing.py first.", preprocessor_src_path)
        sys.exit(1)
    
    preprocessor = load_preprocessor(preprocessor_src_path)

    # -- Step 3.2: Load train / val / test splits -------------------
    logger.info("--- Step 3.2: Loading data splits ---")
    splits_dir = Path(config["paths"]["splits"])

    X_train_raw, y_train = load_split(splits_dir / "train.csv", config)
    X_val_raw, y_val = load_split(splits_dir / "val.csv", config)
    X_test_raw, y_test = load_split(splits_dir / "test.csv", config)

    # -- Step 3.3: Transform features ------------------------------
    logger.info("--- Step 3.3: Transforming features ---")
    X_train = transform_features(X_train_raw, preprocessor)
    X_val = transform_features(X_val_raw, preprocessor)
    X_test = transform_features(X_test_raw, preprocessor)

    feature_names = get_feature_names(preprocessor)

    # -- Step 3.4: Build and train XGBoost -------------------------
    from src.training.train_model import build_xgb_classifier, save_model, train

    logger.info("--- Step 3.4: Training XGBoost ---")
    model = build_xgb_classifier(config)
    model = train(model, X_train, y_train, X_val, y_val)

    # -- Step 3.5: Tune decision threshold on validation -----------
    from src.training.threshold_tuning import (
        apply_threshold,
        find_optimal_threshold,
        save_threshold_to_config,
    )

    logger.info("--- Step 3.5: Tuning decision threshold ---")
    train_cfg = config["training"]
    strategy = train_cfg.get("threshold_strategy", "f1")
    min_precision = train_cfg.get("min_precision", 0.3)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(
        y_val, y_val_proba, strategy=strategy, min_precision=min_precision
    )
    save_threshold_to_config(optimal_threshold, config_path)

    # Log threshold to MLflow
    tracker.log_threshold(optimal_threshold, strategy)

    # -- Step 3.6: Evaluate on all splits --------------------------
    from src.training.evaluate import compute_metrics, generate_all_plots, log_metrics

    logger.info("--- Step 3.6: Evaluating model ---")
    
    eval_dir = PROJECT_ROOT / "metrics" / "runs" / model_version
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Train metrics
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = apply_threshold(y_train_proba, optimal_threshold)
    metrics_train = compute_metrics(y_train, y_train_pred, y_train_proba)
    log_metrics(metrics_train, "Train")
    tracker.log_metrics(metrics_train, prefix="train")

    # Validation metrics
    y_val_pred = apply_threshold(y_val_proba, optimal_threshold)
    metrics_val = compute_metrics(y_val, y_val_pred, y_val_proba)
    log_metrics(metrics_val, "Validation")
    tracker.log_metrics(metrics_val, prefix="val")

    # Test metrics
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = apply_threshold(y_test_proba, optimal_threshold)
    metrics_test = compute_metrics(y_test, y_test_pred, y_test_proba)
    log_metrics(metrics_test, "Test")
    tracker.log_metrics(metrics_test, prefix="test")

    # -- Step 3.7: Generate diagnostic plots -----------------------
    logger.info("--- Step 3.7: Generating evaluation plots ---")
    generate_all_plots(
        y_true=y_test,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        model=model,
        feature_names=feature_names,
        output_dir=eval_dir,
        split_name="test",
    )

    # Log plots to MLflow
    tracker.log_artifacts_dir(eval_dir, artifact_path="plots")

    # -- Step 3.8: Save model and preprocessor to versioned dir ----
    logger.info("--- Step 3.8: Saving model artifacts ---")
    model_dir = Path(config["paths"]["model_artifacts"]) / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "xgb_model.json"
    save_model(model, model_path)
    
    import joblib
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")

    # Update config.yaml with the new model version
    if "api" not in config:
        config["api"] = {}
    config["api"]["model_version"] = model_version
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Log model to MLflow
    tracker.log_artifact(model_path)

    # -- Step 3.9: Generate model card -----------------------------
    from src.training.model_card import generate_model_card

    logger.info("--- Step 3.9: Generating model card ---")
    model_card_dir = Path(config["paths"]["model_cards"])
    model_card_path = model_card_dir / f"model_card_{model_version}.md"
    generate_model_card(
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        config=config,
        output_path=model_card_path,
        threshold=optimal_threshold,
    )

    # Log model card to MLflow
    tracker.log_artifact(model_card_path)

    # -- Step 3.10: Save metrics to history store ------------------
    from src.tracking.metrics_store import save_run_metrics

    logger.info("--- Step 3.10: Saving metrics to history store ---")
    metrics_path = PROJECT_ROOT / "metrics" / "metrics_history.json"
    save_run_metrics(
        config=config,
        threshold=optimal_threshold,
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        version=model_version,
        path=metrics_path,
    )

    # -- Step 3.11: End MLflow run ---------------------------------
    tracker.end_run()

    # -- Done ------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3 COMPLETE")
    logger.info("  Threshold strategy:   %s", strategy)
    logger.info("  Optimal threshold:    %.2f", optimal_threshold)
    logger.info("  Test Precision:       %.4f", metrics_test["precision"])
    logger.info("  Test Recall:          %.4f", metrics_test["recall"])
    logger.info("  Test F1:              %.4f", metrics_test["f1"])
    logger.info("  Test ROC-AUC:         %.4f", metrics_test["roc_auc"])
    logger.info("  Test PR-AUC:          %.4f", metrics_test["pr_auc"])
    logger.info("  Model saved to:       %s", model_path)
    logger.info("  Plots saved to:       %s", eval_dir)
    logger.info("  Model card:           %s", model_card_path)
    logger.info("  Metrics store:        %s", metrics_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
