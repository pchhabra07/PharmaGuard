"""
PharmaGuard — FAERS ZIP Extraction & Table Loading
====================================================
Extracts downloaded FAERS ZIP archives and loads the
dollar-delimited ASCII tables into Pandas DataFrames.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# FAERS ASCII files use '$' as delimiter
FAERS_DELIMITER = "$"


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """
    Extract a FAERS ZIP archive to a destination directory.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP file.
    dest_dir : Path
        Directory to extract contents into.

    Returns
    -------
    Path
        Path to the extracted directory (the top-level folder inside the ZIP).
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting: %s -> %s", zip_path.name, dest_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # FAERS ZIPs typically contain a single top-level directory
    extracted_dirs = [d for d in dest_dir.iterdir() if d.is_dir()]
    if extracted_dirs:
        extracted_dir = extracted_dirs[0]
    else:
        extracted_dir = dest_dir

    logger.info("Extraction complete: %s", extracted_dir)
    return extracted_dir


def find_table_file(extracted_dir: Path, table_name: str) -> Path:
    """
    Find the FAERS table file inside the extracted directory.

    FAERS files are named like DEMO25Q1.txt, DRUG25Q1.txt, etc.
    The naming convention varies, so we search case-insensitively.

    Parameters
    ----------
    extracted_dir : Path
        Directory containing extracted FAERS files.
    table_name : str
        Table name to find (e.g. 'DEMO', 'DRUG', 'REAC', 'OUTC', 'INDI').

    Returns
    -------
    Path
        Path to the matching file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    pattern = table_name.upper()
    candidates = []

    for f in extracted_dir.rglob("*"):
        if f.is_file() and f.name.upper().startswith(pattern) and f.suffix.lower() == ".txt":
            candidates.append(f)

    if not candidates:
        raise FileNotFoundError(
            f"No file matching table '{table_name}' found in {extracted_dir}. "
            f"Available files: {[f.name for f in extracted_dir.rglob('*.txt')]}"
        )

    # Pick the largest file if multiple matches (handles duplicates)
    selected = max(candidates, key=lambda f: f.stat().st_size)
    logger.info("Found table file for %s: %s", table_name, selected.name)
    return selected


def load_faers_table(extracted_dir: Path, table_name: str) -> pd.DataFrame:
    """
    Load a single FAERS ASCII table into a Pandas DataFrame.

    Parameters
    ----------
    extracted_dir : Path
        Directory containing extracted FAERS files.
    table_name : str
        One of: DEMO, DRUG, REAC, OUTC, INDI.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    file_path = find_table_file(extracted_dir, table_name)

    logger.info("Loading table %s from %s", table_name, file_path.name)

    # FAERS uses '$' delimiter, latin-1 encoding handles special chars
    df = pd.read_csv(
        file_path,
        sep=FAERS_DELIMITER,
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="warn",
    )

    # Normalize column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()

    logger.info(
        "Loaded %s: %d rows x %d columns - columns: %s",
        table_name,
        len(df),
        len(df.columns),
        list(df.columns),
    )
    return df


def load_all_tables(extracted_dir: Path, table_names: list) -> dict:
    """
    Load all specified FAERS tables into a dictionary.

    Parameters
    ----------
    extracted_dir : Path
        Directory containing extracted FAERS files.
    table_names : list[str]
        List of table names to load (e.g. ['DEMO', 'DRUG', 'REAC', 'OUTC', 'INDI']).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping table name → DataFrame.
    """
    tables = {}
    for name in table_names:
        tables[name] = load_faers_table(extracted_dir, name)

    logger.info("All %d tables loaded successfully.", len(tables))
    return tables
