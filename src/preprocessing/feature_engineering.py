"""
PharmaGuard — Feature Engineering
====================================
Engineers features from the joined FAERS dataset:
  - polypharmacy_count (number of concurrent drugs per patient)
  - age_group (clinical age buckets)
  - report_quarter (temporal feature)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_polypharmacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the polypharmacy count: how many drugs each patient is taking.

    Groups by primaryid in the DataFrame and counts unique drug names.

    Parameters
    ----------
    df : pd.DataFrame
        Joined FAERS DataFrame with 'primaryid' and a drug name column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'polypharmacy_count' column added.
    """
    df = df.copy()

    drug_col = "drug_name_normalized" if "drug_name_normalized" in df.columns else "drugname"

    drug_counts = (
        df.groupby("primaryid")[drug_col]
        .nunique()
        .reset_index()
        .rename(columns={drug_col: "polypharmacy_count"})
    )

    # Drop existing column if present to avoid merge conflicts
    if "polypharmacy_count" in df.columns:
        df = df.drop(columns=["polypharmacy_count"])

    df = pd.merge(df, drug_counts, on="primaryid", how="left")
    df["polypharmacy_count"] = df["polypharmacy_count"].fillna(1).astype(int)

    logger.info(
        "Polypharmacy computed - mean: %.1f, median: %d, max: %d",
        df["polypharmacy_count"].mean(),
        df["polypharmacy_count"].median(),
        df["polypharmacy_count"].max(),
    )

    return df


def bucket_age(df: pd.DataFrame, bins: list, labels: list, unknown_label: str = "unknown") -> pd.DataFrame:
    """
    Bucket continuous age into clinical age groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'age' column (or 'age_in_years').
    bins : list
        Bin edges for age groups (e.g. [0, 18, 45, 65, 75, 120]).
    labels : list
        Labels for each bin (e.g. ['0-17', '18-44', '45-64', '65-74', '75+']).
    unknown_label : str
        Label for missing or out-of-range ages.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'age_group' column added.
    """
    df = df.copy()

    # Find the age column
    age_col = None
    for candidate in ["age", "age_in_years"]:
        if candidate in df.columns:
            age_col = candidate
            break

    if age_col is None:
        logger.warning("No age column found. Creating 'age_group' as all '%s'.", unknown_label)
        df["age_group"] = unknown_label
        return df

    # Convert to numeric, coercing errors
    age_numeric = pd.to_numeric(df[age_col], errors="coerce")

    # Some FAERS records have age_cod indicating the unit (YR, MON, DEC, etc.)
    if "age_cod" in df.columns:
        age_cod = df["age_cod"].astype(str).str.strip().str.upper()

        # Convert decades to years
        decade_mask = age_cod == "DEC"
        age_numeric = age_numeric.where(~decade_mask, age_numeric * 10)

        # Convert months to years
        month_mask = age_cod == "MON"
        age_numeric = age_numeric.where(~month_mask, age_numeric / 12)

        # Convert days to years
        day_mask = age_cod == "DY"
        age_numeric = age_numeric.where(~day_mask, age_numeric / 365)

        # Convert hours to years
        hour_mask = age_cod == "HR"
        age_numeric = age_numeric.where(~hour_mask, age_numeric / (365 * 24))

    # Filter to reasonable range
    age_numeric = age_numeric.where((age_numeric >= 0) & (age_numeric <= 120))

    # Bucket into groups
    df["age_group"] = pd.cut(age_numeric, bins=bins, labels=labels, right=False).astype(str)

    # Fill missing / out-of-range with unknown
    df["age_group"] = df["age_group"].replace("nan", unknown_label)
    df["age_group"] = df["age_group"].fillna(unknown_label)

    logger.info("Age groups:\n%s", df["age_group"].value_counts().to_string())

    return df


def add_report_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the reporting quarter from the event date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'event_dt' or 'init_fda_dt' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'report_quarter' column added.
    """
    df = df.copy()

    # Try different date columns
    date_col = None
    for candidate in ["event_dt", "init_fda_dt", "fda_dt"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        logger.warning("No date column found. Setting 'report_quarter' to 'unknown'.")
        df["report_quarter"] = "unknown"
        return df

    # Parse date — FAERS dates are typically YYYYMMDD format
    date_parsed = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")

    df["report_quarter"] = (
        date_parsed.dt.year.astype(str) + "-Q" + date_parsed.dt.quarter.astype(str)
    )
    df["report_quarter"] = df["report_quarter"].fillna("unknown")

    logger.info("Report quarters (top 5):\n%s", df["report_quarter"].value_counts().head().to_string())

    return df


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Joined FAERS DataFrame (after drug normalization).
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features.
    """
    logger.info("Starting feature engineering on %d rows...", len(df))

    # Step 2.2: Polypharmacy count
    df = compute_polypharmacy(df)

    # Step 2.3: Age bucketing
    bins = config["preprocessing"]["age_bins"]
    labels = config["preprocessing"]["age_labels"]
    unknown_label = config["preprocessing"]["unknown_age_label"]
    df = bucket_age(df, bins, labels, unknown_label)

    # Step 2.3b: Report quarter
    df = add_report_quarter(df)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df
