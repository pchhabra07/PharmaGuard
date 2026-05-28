"""
PharmaGuard — FAERS Table Joining Module
==========================================
Merges the 5 FAERS relational tables into a single unified
DataFrame and derives the binary 'is_serious' target label.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def derive_target(outc_df: pd.DataFrame, serious_codes: list) -> pd.DataFrame:
    """
    Create a binary 'is_serious' label from the OUTC (outcome) table.

    A case is considered serious if ANY of its outcome codes
    matches a serious code (DE=death, HO=hospitalization,
    DS=disability, LT=life-threatening, OT=other serious).

    Parameters
    ----------
    outc_df : pd.DataFrame
        The OUTC table with columns including 'primaryid' and 'outc_cod'.
    serious_codes : list[str]
        List of outcome codes considered serious.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['primaryid', 'is_serious'].
    """
    outc = outc_df.copy()

    # Normalize outcome code
    outc["outc_cod"] = outc["outc_cod"].astype(str).str.strip().str.upper()

    # Flag serious outcomes
    outc["is_serious_flag"] = outc["outc_cod"].isin(serious_codes).astype(int)

    # Aggregate per case: if ANY outcome is serious, label = 1
    target = (
        outc.groupby("primaryid", as_index=False)["is_serious_flag"]
        .max()
        .rename(columns={"is_serious_flag": "is_serious"})
    )

    logger.info(
        "Target derived - Serious: %d (%.1f%%), Not serious: %d (%.1f%%)",
        target["is_serious"].sum(),
        target["is_serious"].mean() * 100,
        (target["is_serious"] == 0).sum(),
        (1 - target["is_serious"].mean()) * 100,
    )
    return target


def join_tables(tables: dict, serious_codes: list) -> pd.DataFrame:
    """
    Join FAERS tables into a single unified DataFrame.

    Join order:
        DEMO (demographics) ← DRUG (drug info) ← OUTC-derived target

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary of loaded FAERS tables (keys: DEMO, DRUG, OUTC, etc.).
    serious_codes : list[str]
        Outcome codes considered serious.

    Returns
    -------
    pd.DataFrame
        Unified DataFrame with one row per patient-drug report.
    """
    demo = tables["DEMO"].copy()
    drug = tables["DRUG"].copy()
    outc = tables["OUTC"].copy()

    # Derive target label
    target = derive_target(outc, serious_codes)

    logger.info("Joining DEMO (%d rows) <- DRUG (%d rows)...", len(demo), len(drug))

    # Join DEMO and DRUG on primaryid
    merged = pd.merge(demo, drug, on="primaryid", how="inner", suffixes=("", "_drug"))

    logger.info("After DEMO <- DRUG join: %d rows", len(merged))

    # Join with target labels
    merged = pd.merge(merged, target, on="primaryid", how="left")

    # Cases without outcome data: label as not serious (conservative)
    merged["is_serious"] = merged["is_serious"].fillna(0).astype(int)

    logger.info("After target join: %d rows", len(merged))
    logger.info(
        "Final class distribution - Serious: %d (%.1f%%), Not serious: %d (%.1f%%)",
        merged["is_serious"].sum(),
        merged["is_serious"].mean() * 100,
        (merged["is_serious"] == 0).sum(),
        (1 - merged["is_serious"].mean()) * 100,
    )

    return merged


def save_joined(df: pd.DataFrame, dest_dir: Path, filename: str = "faers_joined.parquet") -> Path:
    """
    Save the joined DataFrame to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Joined FAERS DataFrame.
    dest_dir : Path
        Destination directory.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Path to the saved file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / filename
    df.to_parquet(dest_path, index=False)

    logger.info("Saved joined data: %s (%.1f MB, %d rows)", dest_path, dest_path.stat().st_size / 1e6, len(df))
    return dest_path
