"""
PharmaGuard — Class Imbalance Weight Computation
===================================================
Computes the scale_pos_weight parameter for XGBoost:
    scale_pos_weight = count(negative) / count(positive)

This value is written back to config.yaml for use in training.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute the XGBoost scale_pos_weight from target labels.

    Parameters
    ----------
    y : pd.Series
        Binary target labels (0 = not serious, 1 = serious).

    Returns
    -------
    float
        Ratio of negative to positive samples, rounded to 2 decimals.
    """
    counts = y.value_counts()

    n_negative = counts.get(0, 0)
    n_positive = counts.get(1, 0)

    if n_positive == 0:
        logger.error("No positive samples found! Cannot compute scale_pos_weight.")
        raise ValueError("No positive samples (is_serious=1) found in the target column.")

    weight = float(round(n_negative / n_positive, 2))

    logger.info(
        "Class counts - Negative: %d, Positive: %d, scale_pos_weight: %.2f",
        n_negative,
        n_positive,
        weight,
    )

    return weight


def save_to_config(weight: float, config_path: Path) -> None:
    """
    Write the computed scale_pos_weight back to config.yaml.

    Parameters
    ----------
    weight : float
        Computed scale_pos_weight value.
    config_path : Path
        Path to config.yaml.
    """
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["training"]["scale_pos_weight"] = weight

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Updated config.yaml - training.scale_pos_weight = %.2f", weight)
