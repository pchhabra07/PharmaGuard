"""
PharmaGuard — Exploratory Data Analysis (EDA)
================================================
Generates and saves visualizations for the joined FAERS dataset.
All plots are saved as PNG files to the configured EDA directory.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

# ── Plot style ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DPI = 150
FIGSIZE = (10, 6)


def plot_class_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Bar chart of target class distribution (serious vs not serious)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["is_serious"].value_counts().sort_index()
    labels = ["Not Serious (0)", "Serious (1)"]
    colors = ["#4CAF50", "#F44336"]

    ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)

    for i, v in enumerate(counts.values):
        pct = v / len(df) * 100
        ax.text(i, v + len(df) * 0.01, f"{v:,}\n({pct:.1f}%)", ha="center", fontweight="bold")

    ax.set_title("Class Distribution - Serious vs Not Serious ADRs", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Reports")
    ax.set_xlabel("Outcome Class")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_missing_values(df: pd.DataFrame, save_path: Path) -> None:
    """Horizontal bar chart of missing value percentage per column."""
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=True)
    null_pct = null_pct[null_pct > 0]  # only show columns with missing data

    if null_pct.empty:
        logger.info("No missing values found - skipping missing values plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(null_pct) * 0.4)))

    ax.barh(null_pct.index, null_pct.values, color="#FF9800", edgecolor="black", linewidth=0.5)

    for i, v in enumerate(null_pct.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

    ax.set_title("Missing Values by Column (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Missing (%)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_age_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Histogram of patient age values."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Try to extract numeric age
    age_col = None
    for candidate in ["age", "age_cod", "age_in_years"]:
        if candidate in df.columns:
            age_col = candidate
            break

    if age_col is None:
        logger.warning("No age column found - skipping age distribution plot.")
        return

    age_data = pd.to_numeric(df[age_col], errors="coerce").dropna()
    # Filter to reasonable range
    age_data = age_data[(age_data >= 0) & (age_data <= 120)]

    if age_data.empty:
        logger.warning("No valid age data - skipping age distribution plot.")
        return

    ax.hist(age_data, bins=50, color="#2196F3", edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_title("Patient Age Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Reports")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_sex_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Count plot of patient sex."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "sex" not in df.columns:
        logger.warning("No 'sex' column - skipping sex distribution plot.")
        return

    sex_counts = df["sex"].fillna("Unknown").value_counts()
    colors = ["#2196F3", "#E91E63", "#9E9E9E", "#FF9800"]

    ax.bar(sex_counts.index, sex_counts.values, color=colors[: len(sex_counts)], edgecolor="black", linewidth=0.5)

    for i, v in enumerate(sex_counts.values):
        ax.text(i, v + len(df) * 0.005, f"{v:,}", ha="center", fontweight="bold")

    ax.set_title("Patient Sex Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Reports")
    ax.set_xlabel("Sex")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_top_drugs(df: pd.DataFrame, save_path: Path, n: int = 20) -> None:
    """Horizontal bar chart of the most frequently reported drugs."""
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))

    if "drugname" not in df.columns:
        logger.warning("No 'drugname' column - skipping top drugs plot.")
        return

    drug_counts = df["drugname"].fillna("UNKNOWN").str.upper().value_counts().head(n)
    drug_counts = drug_counts.sort_values(ascending=True)

    ax.barh(drug_counts.index, drug_counts.values, color="#673AB7", edgecolor="black", linewidth=0.5)

    for i, v in enumerate(drug_counts.values):
        ax.text(v + drug_counts.max() * 0.01, i, f"{v:,}", va="center")

    ax.set_title(f"Top {n} Most Frequently Reported Drugs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Reports")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Run all EDA visualizations and save PNGs.

    Parameters
    ----------
    df : pd.DataFrame
        Joined FAERS DataFrame.
    output_dir : str | Path
        Directory to save all PNG plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running EDA - saving plots to %s", output_dir)
    logger.info("Dataset shape: %d rows x %d columns", len(df), len(df.columns))

    plot_class_distribution(df, output_dir / "class_distribution.png")
    plot_missing_values(df, output_dir / "missing_values.png")
    plot_age_distribution(df, output_dir / "age_distribution.png")
    plot_sex_distribution(df, output_dir / "sex_distribution.png")
    plot_top_drugs(df, output_dir / "top_drugs.png")

    logger.info("EDA complete - all plots saved to %s", output_dir)
