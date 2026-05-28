"""
PharmaGuard — Drug Name Normalization
=======================================
Uses RapidFuzz fuzzy matching to normalize the hundreds of
spelling variants for the same drug into a canonical form.
"""

import logging
import re
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# Regex to strip dosage/strength info (e.g. '500MG', '10 MG/ML')
DOSAGE_PATTERN = re.compile(
    r"\s*\d+(\.\d+)?\s*(mg|mcg|g|ml|%|iu|units?|meq|mmol)(/\s*(ml|l|g|mg|dose|hr))?",
    re.IGNORECASE,
)

# Regex to strip formulation suffixes
FORMULATION_PATTERN = re.compile(
    r"\s*(tablet|capsule|injection|solution|suspension|cream|ointment|patch|inhaler|syrup|drops?)s?\s*$",
    re.IGNORECASE,
)


def normalize_drug_name(name: str) -> str:
    """
    Clean and normalize a single drug name string.

    Steps:
    1. Convert to lowercase
    2. Strip whitespace
    3. Remove dosage patterns (e.g. '500mg')
    4. Remove formulation suffixes (e.g. 'tablet')
    5. Collapse multiple spaces

    Parameters
    ----------
    name : str
        Raw drug name.

    Returns
    -------
    str
        Cleaned drug name.
    """
    if pd.isna(name) or not isinstance(name, str):
        return "unknown"

    name = name.lower().strip()
    name = DOSAGE_PATTERN.sub("", name)
    name = FORMULATION_PATTERN.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name if name else "unknown"


def build_drug_mapping(drug_series: pd.Series, threshold: int = 85, top_n: int = 500) -> dict:
    """
    Build a fuzzy-match mapping from drug name variants to canonical forms.

    Strategy:
    1. Count all unique drug names (after basic normalization).
    2. Take the top_n most frequent as 'canonical' references.
    3. For each remaining name, find the best fuzzy match among
       the canonical names. If the score >= threshold, map to it.

    Parameters
    ----------
    drug_series : pd.Series
        Series of raw drug names.
    threshold : int
        Minimum fuzzy match score (0-100) to accept a mapping.
    top_n : int
        Number of most-frequent drugs to use as canonical references.

    Returns
    -------
    dict
        Mapping from original drug name → canonical drug name.
    """
    # Basic normalization first
    normalized = drug_series.apply(normalize_drug_name)

    # Count frequencies
    freq = Counter(normalized)
    canonical_names = [name for name, _ in freq.most_common(top_n)]

    logger.info(
        "Building drug mapping: %d unique names, %d canonical references, threshold=%d",
        len(freq),
        len(canonical_names),
        threshold,
    )

    mapping = {}
    all_names = list(freq.keys())

    for name in all_names:
        if name in canonical_names:
            mapping[name] = name
            continue

        # Fuzzy match against canonical names
        match = process.extractOne(name, canonical_names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            mapping[name] = match[0]
        else:
            mapping[name] = name  # keep as-is if no good match

    mapped_count = sum(1 for k, v in mapping.items() if k != v)
    logger.info(
        "Drug mapping complete: %d names mapped to canonical forms, %d kept as-is",
        mapped_count,
        len(mapping) - mapped_count,
    )

    return mapping


def apply_drug_normalization(df: pd.DataFrame, mapping: dict, drug_col: str = "drugname") -> pd.DataFrame:
    """
    Apply drug name normalization to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw drug names.
    mapping : dict
        Mapping from normalized drug name → canonical drug name.
    drug_col : str
        Name of the drug name column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'drug_name_normalized' column.
    """
    df = df.copy()

    # First apply basic normalization, then the fuzzy mapping
    df["drug_name_normalized"] = (
        df[drug_col]
        .apply(normalize_drug_name)
        .map(mapping)
        .fillna(df[drug_col].apply(normalize_drug_name))
    )

    n_unique_before = df[drug_col].nunique()
    n_unique_after = df["drug_name_normalized"].nunique()

    logger.info(
        "Drug normalization applied: %d unique -> %d unique (%.0f%% reduction)",
        n_unique_before,
        n_unique_after,
        (1 - n_unique_after / max(n_unique_before, 1)) * 100,
    )

    return df
