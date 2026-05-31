"""
PharmaGuard — Phase 7: One-Command Retrain & Deploy
=====================================================
Automated retraining script triggered when new FDA FAERS data
is detected (typically after receiving an auto-generated GitHub Issue).

This script performs the ENTIRE pipeline end-to-end:
  1. Detect new quarters available on the FDA server but missing from config
  2. Update config.yaml with the new quarters
  3. Run data ingestion  (run_ingestion.py)
  4. Run preprocessing   (run_preprocessing.py)
  5. Run model training  (run_training.py) — auto-versions to v3, v4, ...
  6. Stage, commit, and push updated artifacts to GitHub
  7. The CI/CD pipeline + cloud deployment picks up changes automatically

Usage:
    python run_retrain.py

Requirements:
    - Local machine with sufficient RAM (several GB for XGBoost training)
    - Virtual environment activated with all dependencies installed
    - Git configured with push access to the remote repository
"""

import logging
import subprocess
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
        logging.FileHandler("retrain.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("PharmaGuard.Retrain")

# ── Project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    """Write updated config back to config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("Updated config.yaml saved.")


def detect_new_quarters(config: dict) -> list:
    """
    Detect FAERS quarters available on the FDA server but not in config.

    Returns
    -------
    list[str]
        List of new quarter strings (e.g. ['2026q2', '2026q3']).
    """
    from src.utils.check_new_data import (
        check_quarter_exists,
        generate_candidate_quarters,
    )

    trained = [q.lower().strip() for q in config["faers"]["quarters"]]
    candidates = generate_candidate_quarters()
    missing = [q for q in candidates if q not in trained]

    base_url = config["faers"]["base_url"]
    new_quarters = []

    logger.info("Checking %d candidate quarter(s) on FDA server...", len(missing))
    for quarter in missing:
        exists = check_quarter_exists(quarter, base_url)
        status = "✅ AVAILABLE" if exists else "—  not yet"
        logger.info("  %s: %s", quarter, status)
        if exists:
            new_quarters.append(quarter)

    return new_quarters


def run_python_script(script_name: str) -> None:
    """
    Run a Python script as a subprocess and stream its output.

    Raises SystemExit if the script fails.
    """
    logger.info("=" * 60)
    logger.info("Running: python %s", script_name)
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, script_name],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        logger.error("❌ %s failed with return code %d", script_name, result.returncode)
        sys.exit(1)

    logger.info("✅ %s completed successfully.", script_name)


def git_stage_commit_push(new_quarters: list, model_version: str) -> None:
    """
    Stage updated artifacts (excluding PNGs), commit, and push to GitHub.

    Parameters
    ----------
    new_quarters : list[str]
        New quarters that were added (for the commit message).
    model_version : str
        The model version string (e.g. 'v3').
    """
    logger.info("=" * 60)
    logger.info("Staging and committing changes to Git...")
    logger.info("=" * 60)

    def run_git(*args):
        cmd = ["git"] + list(args)
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Git command failed: %s", " ".join(cmd))
            logger.error("stderr: %s", result.stderr.strip())
            sys.exit(1)
        return result.stdout.strip()

    # Stage specific files — NOT evaluation PNGs
    files_to_stage = [
        "config.yaml",
        f"model_artifacts/{model_version}/",
        "metrics/metrics_history.json",
        f"model_cards/model_card_{model_version}.md",
    ]

    for f in files_to_stage:
        full_path = PROJECT_ROOT / f
        if full_path.exists():
            run_git("add", f)
            logger.info("  Staged: %s", f)
        else:
            logger.warning("  Skipped (not found): %s", f)

    # Commit
    quarters_str = ", ".join(new_quarters)
    commit_msg = (
        f"chore: retrain model to {model_version} — "
        f"added quarter(s) [{quarters_str}]"
    )
    run_git("commit", "-m", commit_msg)
    logger.info("Committed: %s", commit_msg)

    # Push
    logger.info("Pushing to origin/main...")
    run_git("push", "origin", "main")
    logger.info("✅ Pushed successfully. CI/CD pipeline will trigger automatically.")


def main():
    """Execute the full retrain-and-deploy pipeline."""
    logger.info("=" * 60)
    logger.info("PharmaGuard — Phase 7: Retrain & Deploy")
    logger.info("=" * 60)

    # ── Step 1: Detect new data ────────────────────────────────
    logger.info("--- Step 1: Detecting new FDA FAERS data ---")
    config = load_config()
    new_quarters = detect_new_quarters(config)

    if not new_quarters:
        logger.info("✅ No new data available. Nothing to do.")
        logger.info("   Current quarters: %s", config["faers"]["quarters"])
        sys.exit(0)

    logger.info(
        "🚨 Found %d new quarter(s): %s",
        len(new_quarters),
        new_quarters,
    )

    # ── Step 2: Update config with new quarters ────────────────
    logger.info("--- Step 2: Updating config.yaml ---")
    existing_quarters = config["faers"]["quarters"]
    config["faers"]["quarters"] = existing_quarters + new_quarters
    save_config(config)
    logger.info(
        "   Updated quarters: %s", config["faers"]["quarters"]
    )

    # ── Step 3: Run ingestion ──────────────────────────────────
    logger.info("--- Step 3: Running data ingestion ---")
    run_python_script("run_ingestion.py")

    # ── Step 4: Run preprocessing ──────────────────────────────
    logger.info("--- Step 4: Running preprocessing ---")
    run_python_script("run_preprocessing.py")

    # ── Step 5: Run training ───────────────────────────────────
    logger.info("--- Step 5: Running model training ---")
    run_python_script("run_training.py")

    # ── Step 6: Determine the new model version ────────────────
    # Reload config after training (run_training.py updates it)
    config = load_config()
    model_version = config.get("api", {}).get("model_version", "v3")
    logger.info("--- Step 6: Model version is %s ---", model_version)

    # ── Step 7: Stage, commit, push ────────────────────────────
    logger.info("--- Step 7: Pushing to GitHub ---")
    git_stage_commit_push(new_quarters, model_version)

    # ── Done ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 7 COMPLETE — Retrain & Deploy")
    logger.info("  New quarters added:  %s", new_quarters)
    logger.info("  Model version:       %s", model_version)
    logger.info("  All quarters:        %s", config["faers"]["quarters"])
    logger.info("  Status:              Pushed to GitHub")
    logger.info("  Next:                CI pipeline runs → Render redeploys")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
