"""
PharmaGuard — API Route Handlers
====================================
Defines all HTTP endpoints for the PharmaGuard prediction service.

Routes are registered on a FastAPI ``APIRouter`` and included in the
main application via ``app.include_router()``.

Endpoints
---------
- ``GET /``              — Public HTML dashboard with metrics showcase
- ``GET /health``        — Health check (model load status + timestamp)
- ``GET /model/info``    — Current model metadata
- ``GET /model/history`` — Historical training runs (JSON)
- ``POST /predict``      — Single ADR severity prediction
- ``POST /predict/batch``— Batch predictions (up to 1000 inputs)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Project root (two levels up from this file) ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ────────────────────────────────────────────────────────────────────
# Dashboard
# ────────────────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    """
    Serve the public HTML dashboard.

    Reads the metrics history JSON and renders a server-side HTML page
    showcasing model performance, configuration, training history,
    and API usage examples.
    """
    from src.api.dashboard import render_dashboard

    config = request.app.state.config
    metrics_path = _PROJECT_ROOT / "metrics" / "metrics_history.json"

    # Determine base URL for API links
    base_url = str(request.base_url).rstrip("/")

    html = render_dashboard(config, metrics_path, base_url)
    return HTMLResponse(content=html)


# ────────────────────────────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verify that the API is running and the model is loaded.",
)
async def health_check(request: Request):
    """
    Return API health status.

    Checks whether the XGBoost model and preprocessor are loaded
    in ``app.state``.  Returns ``"healthy"`` if both are available,
    ``"degraded"`` otherwise.
    """
    model_loaded = (
        hasattr(request.app.state, "model")
        and request.app.state.model is not None
    )
    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


# ────────────────────────────────────────────────────────────────────
# Model info
# ────────────────────────────────────────────────────────────────────


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns configuration and metadata for the currently served model.",
)
async def model_info(request: Request):
    """
    Return metadata about the serving model.

    Reads the model's feature count from ``model.n_features_in_``
    and configuration values from ``config.yaml``.
    """
    config = request.app.state.config
    model = request.app.state.model
    train_cfg = config.get("training", {})
    api_cfg = config.get("api", {})

    return ModelInfoResponse(
        model_type="XGBoost (XGBClassifier)",
        n_features=int(model.n_features_in_),
        threshold=float(train_cfg["threshold"]),
        threshold_strategy=train_cfg.get("threshold_strategy", "f1"),
        training_quarters=config.get("faers", {}).get("quarters", []),
        scale_pos_weight=float(train_cfg.get("scale_pos_weight", 1.0)),
        model_version=api_cfg.get("model_version", "v1"),
    )


# ────────────────────────────────────────────────────────────────────
# Historical runs
# ────────────────────────────────────────────────────────────────────


@router.get(
    "/model/history",
    summary="Training history",
    description="Returns historical training run metrics from the metrics store.",
)
async def model_history():
    """
    Return all historical training runs as JSON.

    Reads from ``metrics/metrics_history.json`` and returns the
    full list with a count.
    """
    from src.tracking.metrics_store import load_metrics_history

    metrics_path = _PROJECT_ROOT / "metrics" / "metrics_history.json"
    history = load_metrics_history(metrics_path)

    return {"runs": history, "count": len(history)}


# ────────────────────────────────────────────────────────────────────
# Chart data (for dynamic dashboard)
# ────────────────────────────────────────────────────────────────────


@router.get(
    "/model/charts",
    summary="Chart data for dashboard",
    description="Returns metrics pre-shaped for Chart.js rendering on the dashboard.",
)
async def model_charts():
    """
    Return chart-ready data for the dashboard.

    Reshapes the metrics history into structures that Chart.js can
    directly consume — confusion matrix cells, per-split metric
    comparisons, and historical trend data.
    """
    import json as _json

    metrics_path = _PROJECT_ROOT / "metrics" / "metrics_history.json"
    history = []
    try:
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                history = _json.load(f)
    except Exception:
        pass

    if not history:
        return {"latest": None, "history_trend": [], "runs_count": 0}

    latest = history[-1]
    test_m = latest.get("metrics", {}).get("test", {})
    train_m = latest.get("metrics", {}).get("train", {})
    val_m = latest.get("metrics", {}).get("val", {})

    # Confusion matrix for the latest run (test split)
    cm = test_m.get("confusion_matrix", [[0, 0], [0, 0]])

    # Per-split comparison (radar chart)
    metric_keys = ["precision", "recall", "f1", "accuracy", "specificity", "roc_auc", "pr_auc"]
    splits_comparison = {
        "labels": [k.replace("_", " ").title() for k in metric_keys],
        "train": [train_m.get(k, 0) for k in metric_keys],
        "val": [val_m.get(k, 0) for k in metric_keys],
        "test": [test_m.get(k, 0) for k in metric_keys],
    }

    # Historical trend
    history_trend = []
    for run in history:
        t = run.get("metrics", {}).get("test", {})
        history_trend.append({
            "version": run.get("version", "v1"),
            "timestamp": run.get("timestamp", ""),
            "f1": t.get("f1", 0),
            "roc_auc": t.get("roc_auc", 0),
            "pr_auc": t.get("pr_auc", 0),
            "precision": t.get("precision", 0),
            "recall": t.get("recall", 0),
        })

    return {
        "latest": {
            "version": latest.get("version", "v1"),
            "threshold": latest.get("threshold", 0.5),
            "threshold_strategy": latest.get("threshold_strategy", "f1"),
            "config": latest.get("config", {}),
            "test_metrics": {k: test_m.get(k, 0) for k in metric_keys},
            "confusion_matrix": {
                "tn": cm[0][0], "fp": cm[0][1],
                "fn": cm[1][0], "tp": cm[1][1],
            },
            "splits_comparison": splits_comparison,
        },
        "history_trend": history_trend,
        "runs_count": len(history),
    }


# ────────────────────────────────────────────────────────────────────
# Single prediction
# ────────────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single prediction",
    description="Predict whether a drug-patient combination is likely to cause a serious adverse event.",
)
async def predict(request: Request, data: PredictionRequest):
    """
    Run inference on a single input.

    1. Converts the Pydantic model to a single-row DataFrame.
    2. Transforms features through the fitted ColumnTransformer.
    3. Runs XGBoost ``predict_proba()`` to get the serious-class probability.
    4. Applies the tuned threshold to produce a binary label.
    5. Returns the prediction with probability and threshold.
    """
    try:
        model = request.app.state.model
        preprocessor = request.app.state.preprocessor
        config = request.app.state.config
        threshold = float(config["training"]["threshold"])
        api_cfg = config.get("api", {})

        # Build a one-row DataFrame
        df = pd.DataFrame([data.model_dump()])

        # Transform through the preprocessor
        X = preprocessor.transform(df)

        # Predict
        proba = float(model.predict_proba(X)[0, 1])
        is_serious = proba >= threshold

        return PredictionResponse(
            is_serious=bool(is_serious),
            probability=round(proba, 4),
            threshold=threshold,
            model_version=api_cfg.get("model_version", "v1"),
        )

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


# ────────────────────────────────────────────────────────────────────
# Batch prediction
# ────────────────────────────────────────────────────────────────────


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    description="Predict ADR severity for multiple drug-patient combinations at once (max 1000).",
)
async def predict_batch(request: Request, data: BatchPredictionRequest):
    """
    Run inference on a batch of inputs.

    Converts all inputs to a single DataFrame for efficient batch
    preprocessing and prediction, rather than looping one by one.
    """
    try:
        model = request.app.state.model
        preprocessor = request.app.state.preprocessor
        config = request.app.state.config
        threshold = float(config["training"]["threshold"])
        api_cfg = config.get("api", {})
        model_version = api_cfg.get("model_version", "v1")

        # Build a multi-row DataFrame
        records = [inp.model_dump() for inp in data.inputs]
        df = pd.DataFrame(records)

        # Transform and predict in one batch
        X = preprocessor.transform(df)
        probas = model.predict_proba(X)[:, 1]

        # Build response list
        predictions = []
        for proba in probas:
            proba_f = float(proba)
            predictions.append(
                PredictionResponse(
                    is_serious=bool(proba_f >= threshold),
                    probability=round(proba_f, 4),
                    threshold=threshold,
                    model_version=model_version,
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )

    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {str(exc)}"
        )
