"""
PharmaGuard — Stratified Data Splitting
=========================================
Splits the preprocessed FAERS dataset into train/val/test sets
using stratified sampling to preserve class ratios.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Split data into train/val/test sets with stratification.

    Uses two rounds of train_test_split:
      1. Split into train + remaining
      2. Split remaining into val + test

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed DataFrame.
    target_col : str
        Name of the target column for stratification.
    train_ratio : float
        Proportion for training set.
    val_ratio : float
        Proportion for validation set.
    test_ratio : float
        Proportion for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    logger.info(
        "Splitting %d rows - train: %.0f%%, val: %.0f%%, test: %.0f%%",
        len(df),
        train_ratio * 100,
        val_ratio * 100,
        test_ratio * 100,
    )

    # Round 1: train vs (val + test)
    remaining_ratio = val_ratio + test_ratio
    train_df, remaining_df = train_test_split(
        df,
        test_size=remaining_ratio,
        stratify=df[target_col],
        random_state=random_state,
    )

    # Round 2: val vs test (from remaining)
    test_proportion = test_ratio / remaining_ratio
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=test_proportion,
        stratify=remaining_df[target_col],
        random_state=random_state,
    )

    # Log split details
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos_rate = split_df[target_col].mean() * 100
        logger.info(
            "  %5s: %7d rows (%.1f%% serious)",
            name,
            len(split_df),
            pos_rate,
        )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dest_dir: Path,
) -> dict:
    """
    Save train/val/test splits as CSV files.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames.
    dest_dir : Path
        Destination directory.

    Returns
    -------
    dict[str, Path]
        Mapping of split name → saved file path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = dest_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        paths[name] = path
        logger.info("Saved %s split: %s (%d rows)", name, path, len(split_df))

    return paths
