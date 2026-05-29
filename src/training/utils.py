"""
PharmaGuard — Training Utilities
===================================
Shared helpers for the training phase: loading split data,
applying the fitted preprocessor, and extracting feature names.

These utilities decouple I/O and data preparation from the core
training and evaluation logic, keeping each module focused on a
single responsibility.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def load_split(path: Path, config: dict) -> tuple:
    """
    Load a CSV split file and separate features from the target column.

    Reads the CSV into a DataFrame, identifies the feature columns
    (categorical + numerical) defined in config, and separates them
    from the target column.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the CSV split file (e.g. train.csv).
    config : dict
        Parsed config.yaml dictionary. Must contain:
        - ``preprocessing.categorical_features`` : list of categorical column names
        - ``preprocessing.numerical_features``   : list of numerical column names
        - ``preprocessing.target_column``         : name of the binary target column

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X contains only the feature columns and y is the
        target Series.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    KeyError
        If the target column is not present in the CSV.
    """
    path = Path(path)
    logger.info("Loading split from: %s", path)

    df = pd.read_csv(path)
    logger.info("  Loaded %d rows × %d columns", len(df), len(df.columns))

    target_col = config["preprocessing"]["target_column"]
    cat_features = config["preprocessing"]["categorical_features"]
    num_features = config["preprocessing"]["numerical_features"]

    # Filter to features that actually exist in the DataFrame
    feature_cols = [c for c in cat_features + num_features if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    logger.info(
        "  Features: %d columns, Target: '%s' (%.1f%% positive)",
        len(feature_cols),
        target_col,
        y.mean() * 100,
    )

    return X, y


def load_preprocessor(path: Path) -> ColumnTransformer:
    """
    Load a previously fitted scikit-learn ColumnTransformer from disk.

    Uses joblib to deserialise the preprocessor that was saved during
    Phase 2. The returned object is ready to call ``.transform()`` on
    new DataFrames.

    Parameters
    ----------
    path : Path
        Path to the saved ``.pkl`` file (e.g. ``model_artifacts/preprocessor.pkl``).

    Returns
    -------
    ColumnTransformer
        The fitted preprocessing pipeline.
    """
    preprocessor = joblib.load(path)
    logger.info("Loaded preprocessor from: %s", path)
    return preprocessor


def transform_features(X: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Apply the fitted ColumnTransformer to a feature DataFrame.

    Transforms categorical columns via OneHotEncoding and passes
    numerical columns through unchanged, producing the dense
    numeric matrix expected by XGBoost.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame with columns matching those the preprocessor
        was fitted on (categorical + numerical).
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer from Phase 2.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_samples, n_output_features)`` ready
        for model training or prediction.
    """
    X_transformed = preprocessor.transform(X)
    logger.info("Transformed features: %s → %s", X.shape, X_transformed.shape)
    return X_transformed


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.

    Calls ``get_feature_names_out()`` on the transformer and converts
    the result to a plain Python list. Feature names follow the format
    ``cat__<column>_<category>`` for one-hot encoded features and
    ``num__<column>`` for passthrough numerical features.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer.

    Returns
    -------
    list[str]
        Ordered list of output feature names matching the columns
        of the transformed array.
    """
    try:
        names = list(preprocessor.get_feature_names_out())
        logger.info("Extracted %d feature names from preprocessor", len(names))
        return names
    except AttributeError:
        logger.warning(
            "Could not extract feature names (sklearn < 1.0?). "
            "Returning generic names."
        )
        # Fallback: generate numbered names
        n_features = preprocessor.transform(
            pd.DataFrame(columns=preprocessor.feature_names_in_)
        ).shape[1]
        return [f"feature_{i}" for i in range(n_features)]
