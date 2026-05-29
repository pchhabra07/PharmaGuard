"""
PharmaGuard — XGBoost Model Training
========================================
Constructs, trains, saves, and loads the XGBoost binary classifier
used to predict serious Adverse Drug Reactions (ADRs).

The model uses ``scale_pos_weight`` to handle the class imbalance
(72.6 % positive / 27.4 % negative) and early stopping on a held-out
validation set to prevent overfitting.

Model persistence uses XGBoost's native JSON format (``.json``), which
is portable across platforms, human-readable for auditability, and
recommended by the XGBoost documentation for long-term storage.
"""

import logging
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_xgb_classifier(config: dict) -> XGBClassifier:
    """
    Instantiate an XGBClassifier with hyperparameters from config.

    Reads the ``training`` section of config.yaml and maps each key
    to the corresponding XGBoost constructor argument. Uses the
    ``binary:logistic`` objective and the evaluation metric specified
    in config (defaults to ``aucpr`` for imbalanced problems).

    Parameters
    ----------
    config : dict
        Parsed config.yaml. Must contain a ``training`` key with at
        least ``scale_pos_weight``. Other supported keys:
        ``n_estimators``, ``max_depth``, ``learning_rate``,
        ``random_state``, ``eval_metric``, ``early_stopping_rounds``.

    Returns
    -------
    XGBClassifier
        An unfitted XGBoost classifier ready for ``.fit()``.
    """
    train_cfg = config["training"]

    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=train_cfg.get("scale_pos_weight", 1.0),
        n_estimators=train_cfg.get("n_estimators", 500),
        max_depth=train_cfg.get("max_depth", 6),
        learning_rate=train_cfg.get("learning_rate", 0.1),
        random_state=train_cfg.get("random_state", 42),
        eval_metric=train_cfg.get("eval_metric", "aucpr"),
        early_stopping_rounds=train_cfg.get("early_stopping_rounds", 30),
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=1,
    )

    logger.info(
        "Built XGBClassifier — n_estimators=%d, max_depth=%d, lr=%.3f, "
        "scale_pos_weight=%.2f, eval_metric=%s, early_stopping=%d",
        model.n_estimators,
        model.max_depth,
        model.learning_rate,
        model.scale_pos_weight,
        model.eval_metric,
        model.early_stopping_rounds,
    )

    return model


def train(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """
    Fit the XGBoost model with early stopping on validation data.

    Trains the model on ``(X_train, y_train)`` while monitoring
    performance on ``(X_val, y_val)``. Training stops early if the
    validation metric does not improve for ``early_stopping_rounds``
    consecutive boosting rounds, preventing overfitting.

    Parameters
    ----------
    model : XGBClassifier
        Unfitted classifier (from ``build_xgb_classifier``).
    X_train : np.ndarray
        Transformed training feature matrix.
    y_train : np.ndarray
        Training target labels (0/1).
    X_val : np.ndarray
        Transformed validation feature matrix.
    y_val : np.ndarray
        Validation target labels (0/1).

    Returns
    -------
    XGBClassifier
        The fitted model. ``model.best_iteration`` contains the
        optimal number of boosting rounds.
    """
    logger.info(
        "Training XGBoost — X_train: %s, X_val: %s",
        X_train.shape,
        X_val.shape,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,  # Log every 50 rounds
    )

    best_iter = getattr(model, "best_iteration", model.n_estimators)
    logger.info("Training complete — best iteration: %d", best_iter)

    return model


def save_model(model: XGBClassifier, path: Path) -> Path:
    """
    Save the trained XGBoost model in native JSON format.

    XGBoost's JSON format is:
    - **Portable**: works across operating systems and Python versions
    - **Human-readable**: can be inspected and version-controlled
    - **Recommended**: official XGBoost documentation endorses JSON for
      long-term model storage

    Parameters
    ----------
    model : XGBClassifier
        Fitted XGBoost classifier.
    path : Path
        Destination path (should end in ``.json``).

    Returns
    -------
    Path
        The path where the model was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(path))
    logger.info("Model saved to: %s", path)

    return path


def load_model(path: Path) -> XGBClassifier:
    """
    Load a previously saved XGBoost model from JSON.

    Creates a new ``XGBClassifier`` instance and loads the parameters
    and tree structure from the saved JSON file.

    Parameters
    ----------
    path : Path
        Path to the saved ``.json`` model file.

    Returns
    -------
    XGBClassifier
        The fitted XGBoost classifier, ready for ``.predict()`` and
        ``.predict_proba()``.
    """
    model = XGBClassifier()
    model.load_model(str(path))
    logger.info("Model loaded from: %s", path)

    return model
