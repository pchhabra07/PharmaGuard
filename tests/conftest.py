"""
PharmaGuard — Shared Test Fixtures
======================================
Provides reusable pytest fixtures for the test suite.

Key fixtures:
- ``mock_model``       — A fake XGBoost model that returns fixed probabilities
- ``mock_preprocessor``— A fake sklearn preprocessor that passes data through
- ``test_app``         — A configured FastAPI app with mocked model artifacts
- ``client``           — An ``httpx.AsyncClient`` wired to the test app
- ``sample_config``    — A minimal config.yaml-like dictionary
- ``tmp_metrics_file`` — A temporary metrics JSON file for I/O tests
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app


# ── Project root ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────────────────────
# Sample config (mirrors config.yaml structure)
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_config() -> dict:
    """Minimal config dictionary matching the shape of config.yaml."""
    return {
        "faers": {
            "base_url": "https://fis.fda.gov/content/Exports",
            "quarters": ["2026q1"],
        },
        "paths": {
            "model_artifacts": "model_artifacts",
        },
        "preprocessing": {
            "target_column": "is_serious",
            "categorical_features": ["sex", "route", "age_group", "drug_name_normalized"],
            "numerical_features": ["polypharmacy_count"],
        },
        "training": {
            "scale_pos_weight": 0.38,
            "threshold": 0.24,
            "threshold_strategy": "f1",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 30,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "model_version": "v2",
        },
    }


# ────────────────────────────────────────────────────────────────────
# Mock ML objects
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """
    A mock XGBoost model that returns a fixed probability of 0.85
    (above the default threshold of 0.24, so prediction = serious).
    """
    model = MagicMock()
    model.n_features_in_ = 396
    # predict_proba returns shape (n_samples, 2): [not_serious, serious]
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.array([[0.15, 0.85]] * X.shape[0])
    )
    return model


@pytest.fixture
def mock_preprocessor():
    """
    A mock preprocessor that returns a numpy identity matrix.

    Real preprocessor transforms a DataFrame into a sparse matrix with
    396 columns.  For tests we return a simple (n, 396) array so
    ``predict_proba`` gets the right shape.
    """
    preprocessor = MagicMock()
    preprocessor.transform = MagicMock(
        side_effect=lambda df: np.zeros((len(df), 396))
    )
    return preprocessor


# ────────────────────────────────────────────────────────────────────
# Test FastAPI app & async client
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def test_app(sample_config, mock_model, mock_preprocessor):
    """
    Create a FastAPI app with mocked model artifacts injected into
    ``app.state``, bypassing the lifespan (which loads from disk).
    """
    app = create_app()

    # Inject mocks into app.state directly
    app.state.config = sample_config
    app.state.model = mock_model
    app.state.preprocessor = mock_preprocessor

    return app


@pytest_asyncio.fixture
async def client(test_app):
    """
    An async HTTP client wired to the test app.

    Usage in tests::

        async def test_something(client):
            resp = await client.get("/health")
            assert resp.status_code == 200
    """
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ────────────────────────────────────────────────────────────────────
# Temp metrics file
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_metrics_file(tmp_path) -> Path:
    """Create a temporary metrics_history.json with one sample run."""
    data = [
        {
            "run_id": "run_001_abc123",
            "timestamp": "2026-05-29T15:00:00",
            "config": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": 0.38,
                "eval_metric": "aucpr",
                "early_stopping_rounds": 30,
            },
            "threshold": 0.24,
            "threshold_strategy": "f1",
            "version": "v1",
            "quarters": ["2026q1"],
            "metrics": {
                "train": {
                    "accuracy": 0.85,
                    "precision": 0.78,
                    "recall": 0.91,
                    "f1": 0.84,
                    "roc_auc": 0.92,
                    "pr_auc": 0.88,
                    "specificity": 0.72,
                    "confusion_matrix": [[3600, 1400], [900, 9100]],
                },
                "val": {
                    "accuracy": 0.83,
                    "precision": 0.76,
                    "recall": 0.89,
                    "f1": 0.82,
                    "roc_auc": 0.90,
                    "pr_auc": 0.86,
                    "specificity": 0.70,
                    "confusion_matrix": [[700, 300], [220, 1780]],
                },
                "test": {
                    "accuracy": 0.82,
                    "precision": 0.75,
                    "recall": 0.88,
                    "f1": 0.81,
                    "roc_auc": 0.89,
                    "pr_auc": 0.85,
                    "specificity": 0.69,
                    "confusion_matrix": [[690, 310], [240, 1760]],
                },
            },
        }
    ]

    metrics_path = tmp_path / "metrics_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return metrics_path
