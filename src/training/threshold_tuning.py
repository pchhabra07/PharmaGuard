"""
PharmaGuard — Decision Threshold Tuning
===========================================
Finds the optimal probability threshold for converting XGBoost's
continuous predictions into binary serious / not-serious labels.

The default threshold of 0.5 is rarely optimal for imbalanced
problems. This module scans candidate thresholds and selects the
one that best balances precision and recall according to the
chosen strategy.

Supported strategies
--------------------
- **f1** : Maximise the F1 score (harmonic mean of precision and recall).
- **recall_at_precision** : Maximise recall while maintaining precision
  above a user-defined minimum (useful in medical contexts where
  missing a serious event is costlier than a false alarm).
- **youden** : Maximise Youden's J statistic (TPR − FPR), which finds
  the point on the ROC curve farthest from the random-chance diagonal.
"""

import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
    min_precision: float = 0.3,
) -> float:
    """
    Scan probability thresholds to find the optimal decision boundary.

    Evaluates every threshold from 0.05 to 0.95 in steps of 0.01 and
    selects the best according to the chosen ``strategy``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    strategy : str
        Optimisation strategy. One of:

        - ``"f1"`` — maximise F1 score.
        - ``"recall_at_precision"`` — maximise recall subject to
          ``precision >= min_precision``.
        - ``"youden"`` — maximise TPR − FPR (Youden's J).

    min_precision : float
        Minimum acceptable precision when ``strategy="recall_at_precision"``.
        Ignored for other strategies.

    Returns
    -------
    float
        The optimal threshold, rounded to 2 decimal places.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the supported values.
    """
    if strategy not in ("f1", "recall_at_precision", "youden"):
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: f1, recall_at_precision, youden"
        )

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_score = -1.0

    if strategy == "youden":
        # Use sklearn's ROC curve for exact threshold enumeration
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(roc_thresholds[best_idx])
        best_score = float(j_scores[best_idx])

        logger.info(
            "Youden's J — optimal threshold: %.2f (J=%.4f, TPR=%.4f, FPR=%.4f)",
            best_threshold,
            best_score,
            tpr[best_idx],
            fpr[best_idx],
        )

    elif strategy == "f1":
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        logger.info(
            "F1 strategy — optimal threshold: %.2f (F1=%.4f)",
            best_threshold,
            best_score,
        )

    elif strategy == "recall_at_precision":
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)

            if prec >= min_precision and rec > best_score:
                best_score = rec
                best_threshold = float(t)

        logger.info(
            "Recall-at-precision strategy — optimal threshold: %.2f "
            "(recall=%.4f, min_precision=%.2f)",
            best_threshold,
            best_score,
            min_precision,
        )

    return round(best_threshold, 2)


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert continuous probabilities to binary predictions.

    Applies a simple comparison: predictions above or equal to the
    threshold are classified as positive (serious), otherwise negative.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Decision boundary (typically between 0 and 1).

    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1) with the same shape as ``y_proba``.
    """
    return (y_proba >= threshold).astype(int)


def save_threshold_to_config(threshold: float, config_path: Path) -> None:
    """
    Write the tuned threshold back to config.yaml.

    Reads the existing config, updates ``training.threshold``, and
    writes the file back preserving all other keys. This ensures the
    threshold used during training is persisted for reproducibility
    and used consistently during inference.

    Parameters
    ----------
    threshold : float
        The optimal threshold value.
    config_path : Path
        Path to config.yaml.
    """
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["training"]["threshold"] = threshold

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Updated config.yaml — training.threshold = %.2f", threshold)
