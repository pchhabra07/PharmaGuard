"""
PharmaGuard — Schema Validation Module
=========================================
Validates the structure and quality of the joined FAERS DataFrame.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we expect after joining DEMO + DRUG + target
EXPECTED_COLUMNS = [
    "primaryid",
    "age",
    "sex",
    "drugname",
    "route",
    "is_serious",
]

# Maximum acceptable null rate before logging a warning
NULL_RATE_THRESHOLD = 0.50

# Minimum expected row count (sanity check)
MIN_ROW_COUNT = 1000


def validate_schema(df: pd.DataFrame) -> list:
    """
    Validate the joined FAERS DataFrame against expected schema.

    Checks performed:
    1. Expected columns are present.
    2. No column exceeds the null rate threshold.
    3. Row count is above the minimum expected.

    Parameters
    ----------
    df : pd.DataFrame
        The joined FAERS DataFrame to validate.

    Returns
    -------
    list[str]
        List of warning messages. Empty list = all checks passed.
    """
    warnings_list = []

    # ── Check 1: Expected columns ──────────────────────────────
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
         msg = f"MISSING COLUMNS: {missing_cols}. Available: {list(df.columns)}"
         logger.warning(msg)
         warnings_list.append(msg)
    else:
         logger.info("OK: All expected columns present.")

    # ── Check 2: Null rates ────────────────────────────────────
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > NULL_RATE_THRESHOLD]

    if not high_null_cols.empty:
         for col, rate in high_null_cols.items():
             msg = f"HIGH NULL RATE: column '{col}' has {rate:.1%} nulls (threshold: {NULL_RATE_THRESHOLD:.0%})"
             logger.warning(msg)
             warnings_list.append(msg)
    else:
         logger.info("OK: No columns exceed %.0f%% null rate.", NULL_RATE_THRESHOLD * 100)

    # ── Check 3: Row count ─────────────────────────────────────
    if len(df) < MIN_ROW_COUNT:
         msg = f"LOW ROW COUNT: {len(df)} rows (expected at least {MIN_ROW_COUNT})"
         logger.warning(msg)
         warnings_list.append(msg)
    else:
         logger.info("OK: Row count OK: %d rows.", len(df))

    # ── Check 4: Target distribution ───────────────────────────
    if "is_serious" in df.columns:
         class_counts = df["is_serious"].value_counts()
         logger.info("Target distribution:\n%s", class_counts.to_string())

         if df["is_serious"].nunique() < 2:
             msg = "SINGLE CLASS: target column 'is_serious' has only one unique value."
             logger.warning(msg)
             warnings_list.append(msg)

    # ── Summary ────────────────────────────────────────────────
    if not warnings_list:
         logger.info("OK: Schema validation PASSED — no issues found.")
    else:
         logger.warning("Schema validation completed with %d warning(s).", len(warnings_list))

    return warnings_list
