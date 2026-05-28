"""
PharmaGuard — Scikit-learn Preprocessing Pipeline Builder
==========================================================
Constructs a reusable ColumnTransformer that applies:
  - OneHotEncoding for categorical features
  - Passthrough for numerical features

This pipeline is fitted on training data and saved for consistent
use during both training and inference.
"""

import logging
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def build_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for FAERS features.

    Parameters
    ----------
    categorical_features : list[str]
        Column names to one-hot encode (e.g. ['sex', 'route', 'age_group', 'drug_name_normalized']).
    numerical_features : list[str]
        Column names to pass through unchanged (e.g. ['polypharmacy_count']).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=200),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numerical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "Built preprocessor - categoricals: %s, numericals: %s",
        categorical_features,
        numerical_features,
    )

    return preprocessor


def fit_and_save_pipeline(preprocessor: ColumnTransformer, X_train, save_path: Path) -> Path:
    """
    Fit the preprocessor on training data and save it with joblib.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted preprocessing pipeline.
    X_train : pd.DataFrame
        Training feature DataFrame.
    save_path : Path
        Path to save the fitted pipeline.

    Returns
    -------
    Path
        Path to the saved pipeline file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fitting preprocessor on %d training samples...", len(X_train))
    preprocessor.fit(X_train)

    # Log output shape
    X_transformed = preprocessor.transform(X_train[:1])
    n_features_out = X_transformed.shape[1]
    logger.info("Preprocessor fitted. Output features: %d", n_features_out)

    # Save
    joblib.dump(preprocessor, save_path)
    logger.info("Preprocessor saved to: %s", save_path)

    return save_path


def load_pipeline(load_path: Path) -> ColumnTransformer:
    """
    Load a previously saved preprocessor.

    Parameters
    ----------
    load_path : Path
        Path to the saved pipeline file.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessing pipeline.
    """
    preprocessor = joblib.load(load_path)
    logger.info("Preprocessor loaded from: %s", load_path)
    return preprocessor
