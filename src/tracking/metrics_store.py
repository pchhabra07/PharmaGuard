"""
PharmaGuard — Metrics Store
===============================
Persists training run metrics to a JSON file so they can be
displayed on the public dashboard and compared across runs.

Each training run appends a record containing its timestamp,
hyperparameters, tuned threshold, and evaluation metrics for all
splits.  The API dashboard reads this file to render historical
performance data.

File format::

    [
        {
            "run_id": "run_001",
            "timestamp": "2026-05-29T15:00:00",
            "config": { ... },
            "threshold": 0.24,
            "threshold_strategy": "f1",
            "metrics": {
                "train": { ... },
                "val": { ... },
                "test": { ... }
            }
        },
        ...
    ]
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path for the metrics history file
DEFAULT_METRICS_PATH = Path("metrics/metrics_history.json")


def load_metrics_history(path: Path = None) -> list:
    """
    Load all historical run records from the JSON file.

    If the file does not exist, returns an empty list.

    Parameters
    ----------
    path : Path, optional
        Path to the metrics history JSON.  Defaults to
        ``metrics/metrics_history.json``.

    Returns
    -------
    list[dict]
        List of run records, newest last.
    """
    if path is None:
        path = DEFAULT_METRICS_PATH
    path = Path(path)

    if not path.exists():
        logger.info("No metrics history found at %s — returning empty", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        logger.info("Loaded %d historical runs from %s", len(history), path)
        return history
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Failed to load metrics history: %s", exc)
        return []


def save_run_metrics(
    config: dict,
    threshold: float,
    metrics_train: dict,
    metrics_val: dict,
    metrics_test: dict,
    version: str,
    path: Path = None,
) -> dict:
    """
    Append a new training run's metrics to the history file.

    Creates the file and parent directories if they do not exist.

    Parameters
    ----------
    config : dict
        Parsed config.yaml — the ``training`` section is extracted
        and stored alongside the metrics.
    threshold : float
        The optimal decision threshold used for this run.
    metrics_train : dict
        Training split metrics from ``compute_metrics()``.
    metrics_val : dict
        Validation split metrics.
    metrics_test : dict
        Test split metrics.
    version : str
        The version string for this run (e.g., "v1", "v2").
    path : Path, optional
        Path to the metrics history JSON.

    Returns
    -------
    dict
        The newly created run record.
    """
    if path is None:
        path = DEFAULT_METRICS_PATH
    path = Path(path)

    # Load existing history
    history = load_metrics_history(path)

    # Build the run record
    train_cfg = config.get("training", {})
    run_record = {
        "run_id": f"run_{len(history) + 1:03d}_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "n_estimators": train_cfg.get("n_estimators"),
            "max_depth": train_cfg.get("max_depth"),
            "learning_rate": train_cfg.get("learning_rate"),
            "scale_pos_weight": train_cfg.get("scale_pos_weight"),
            "eval_metric": train_cfg.get("eval_metric"),
            "early_stopping_rounds": train_cfg.get("early_stopping_rounds"),
        },
        "threshold": threshold,
        "threshold_strategy": train_cfg.get("threshold_strategy", "f1"),
        "version": version,
        "quarters": config.get("faers", {}).get("quarters", []),
        "metrics": {
            "train": _clean_metrics(metrics_train),
            "val": _clean_metrics(metrics_val),
            "test": _clean_metrics(metrics_test),
        },
    }

    # Append and save
    history.append(run_record)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info("Saved run %s to metrics store (%s)", run_record["run_id"], path)
    return run_record


def get_latest_run(path: Path = None) -> dict:
    """
    Retrieve the most recent run record.

    Parameters
    ----------
    path : Path, optional
        Path to the metrics history JSON.

    Returns
    -------
    dict or None
        The latest run record, or ``None`` if no runs exist.
    """
    history = load_metrics_history(path)
    if not history:
        return None
    return history[-1]


def _clean_metrics(metrics: dict) -> dict:
    """
    Remove non-JSON-serialisable entries from a metrics dict.

    Keeps only scalar (int/float) values and converts numpy types
    to native Python types for JSON compatibility.

    Parameters
    ----------
    metrics : dict
        Raw metrics dictionary from ``compute_metrics()``.

    Returns
    -------
    dict
        Cleaned metrics with only serialisable values.
    """
    cleaned = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            cleaned[key] = value  # Already a nested list
        elif isinstance(value, (int, float)):
            cleaned[key] = round(float(value), 4)
        else:
            try:
                cleaned[key] = round(float(value), 4)
            except (TypeError, ValueError):
                continue
    return cleaned
