"""
PharmaGuard — Model Evaluation
==================================
Computes classification metrics and generates diagnostic visualisations
for the trained XGBoost adverse-event classifier.

All plots are saved as PNG files for inclusion in model cards and
documentation. Metrics are returned as plain dictionaries to keep
this module decoupled from any logging or serialisation choices.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── Consistent plot styling ───────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ────────────────────────────────────────────────────────────────────
# Metrics computation
# ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute a comprehensive set of binary classification metrics.

    Calculates both threshold-dependent metrics (precision, recall,
    F1, accuracy, specificity) and threshold-independent metrics
    (ROC-AUC, PR-AUC) to give a complete picture of model performance
    on imbalanced data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = not serious, 1 = serious).
    y_pred : np.ndarray
        Predicted binary labels after threshold application.
    y_proba : np.ndarray
        Predicted probabilities for the positive class (serious).

    Returns
    -------
    dict
        Dictionary with keys: ``precision``, ``recall``, ``f1``,
        ``accuracy``, ``specificity``, ``roc_auc``, ``pr_auc``,
        ``confusion_matrix`` (as a 2×2 nested list).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def log_metrics(metrics: dict, split_name: str) -> None:
    """
    Pretty-log all evaluation metrics at INFO level.

    Formats each metric on its own line with consistent alignment
    for easy scanning in terminal and log files.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from ``compute_metrics()``.
    split_name : str
        Name of the data split (e.g. ``"Train"``, ``"Val"``, ``"Test"``).
    """
    logger.info("-- %s Metrics --", split_name)
    logger.info("  Precision:    %.4f", metrics["precision"])
    logger.info("  Recall:       %.4f", metrics["recall"])
    logger.info("  F1 Score:     %.4f", metrics["f1"])
    logger.info("  Accuracy:     %.4f", metrics["accuracy"])
    logger.info("  Specificity:  %.4f", metrics["specificity"])
    logger.info("  ROC-AUC:      %.4f", metrics["roc_auc"])
    logger.info("  PR-AUC:       %.4f", metrics["pr_auc"])
    logger.info("  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    logger.info("    TN=%d  FP=%d", cm[0][0], cm[0][1])
    logger.info("    FN=%d  TP=%d", cm[1][0], cm[1][1])


# ────────────────────────────────────────────────────────────────────
# Visualisation functions
# ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> Path:
    """
    Generate and save a colour-coded confusion matrix heatmap.

    Uses seaborn's ``heatmap`` with annotated counts and percentage
    labels inside each cell. The colour scale uses ``Blues`` for
    intuitive darker-is-more visual encoding.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    save_path : Path
        File path to save the PNG image.

    Returns
    -------
    Path
        The path where the plot was saved.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Serious", "Serious"],
        yticklabels=["Not Serious", "Serious"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Adverse Event Severity")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    logger.info("Confusion matrix plot saved to: %s", save_path)
    return save_path


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> Path:
    """
    Generate and save a Receiver Operating Characteristic (ROC) curve.

    Plots the True Positive Rate vs False Positive Rate across all
    probability thresholds, with a dashed diagonal line representing
    a random classifier. The Area Under the Curve (AUC) is annotated
    in the legend.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    save_path : Path
        File path to save the PNG image.

    Returns
    -------
    Path
        The path where the plot was saved.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Adverse Event Classifier")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    logger.info("ROC curve saved to: %s (AUC=%.4f)", save_path, roc_auc)
    return save_path


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> Path:
    """
    Generate and save a Precision-Recall (PR) curve.

    PR curves are more informative than ROC curves for imbalanced
    datasets because they focus on the minority class performance.
    The Average Precision (AP) is annotated in the legend.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    save_path : Path
        File path to save the PNG image.

    Returns
    -------
    Path
        The path where the plot was saved.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=y_true.mean(), color="gray", lw=1, linestyle="--", label=f"Baseline ({y_true.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Adverse Event Classifier")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    logger.info("PR curve saved to: %s (AP=%.4f)", save_path, ap)
    return save_path


def plot_feature_importance(
    model,
    feature_names: list,
    save_path: Path,
    top_n: int = 20,
) -> Path:
    """
    Generate and save a horizontal bar chart of feature importances.

    Extracts the ``feature_importances_`` array from the trained
    XGBoost model, sorts by importance, and plots the top N features
    as a horizontal bar chart for readability.

    Parameters
    ----------
    model : XGBClassifier
        Fitted XGBoost model with ``feature_importances_`` attribute.
    feature_names : list[str]
        Human-readable names corresponding to the model's input features.
    save_path : Path
        File path to save the PNG image.
    top_n : int, optional
        Number of top features to display. Defaults to 20.

    Returns
    -------
    Path
        The path where the plot was saved.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_

    # Truncate feature_names if they don't match (safety)
    n_features = len(importances)
    if len(feature_names) != n_features:
        logger.warning(
            "Feature name count (%d) != importance count (%d). Using indices.",
            len(feature_names),
            n_features,
        )
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Sort and take top N
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(len(top_names)), top_importances[::-1], color="#FF9800", edgecolor="#E65100")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    logger.info("Feature importance plot saved to: %s", save_path)
    return save_path


def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model,
    feature_names: list,
    output_dir: Path,
    split_name: str = "test",
) -> dict:
    """
    Convenience wrapper that generates all four diagnostic plots.

    Calls ``plot_confusion_matrix``, ``plot_roc_curve``,
    ``plot_precision_recall_curve``, and ``plot_feature_importance``
    in sequence, saving each to the specified output directory with
    consistent filenames.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    model : XGBClassifier
        Fitted XGBoost model.
    feature_names : list[str]
        Human-readable feature names.
    output_dir : Path
        Directory to save all plots.
    split_name : str, optional
        Name prefix for plot filenames (default ``"test"``).

    Returns
    -------
    dict[str, Path]
        Mapping of plot name → saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    paths["confusion_matrix"] = plot_confusion_matrix(
        y_true, y_pred, output_dir / f"{split_name}_confusion_matrix.png"
    )
    paths["roc_curve"] = plot_roc_curve(
        y_true, y_proba, output_dir / f"{split_name}_roc_curve.png"
    )
    paths["pr_curve"] = plot_precision_recall_curve(
        y_true, y_proba, output_dir / f"{split_name}_pr_curve.png"
    )
    paths["feature_importance"] = plot_feature_importance(
        model, feature_names, output_dir / f"{split_name}_feature_importance.png"
    )

    logger.info("All %s plots saved to: %s", split_name, output_dir)
    return paths
