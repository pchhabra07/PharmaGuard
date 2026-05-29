"""
PharmaGuard — FastAPI Application Factory
=============================================
Creates and configures the FastAPI application instance.

On startup, the **lifespan** context manager loads:
1. ``config.yaml`` — centralised configuration
2. ``preprocessor.pkl`` — fitted scikit-learn ColumnTransformer
3. ``xgb_model.json`` — trained XGBoost classifier

These are cached in ``app.state`` so every request handler can
access them without re-loading from disk.

The application includes CORS middleware for cross-origin requests
(needed when the dashboard or Swagger UI is accessed from a
different domain).
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ── Project root (two levels up from this file) ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager — loads model artifacts on startup.

    This is called once when the server starts and again (for cleanup)
    when it shuts down.  All heavy objects are attached to ``app.state``
    so they persist for the lifetime of the process.
    """
    # ── Load config ────────────────────────────────────────────
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    app.state.config = config

    # ── Load preprocessor ──────────────────────────────────────
    api_cfg = config.get("api", {})
    model_version = api_cfg.get("model_version", "v1")
    model_dir = _PROJECT_ROOT / config["paths"]["model_artifacts"] / model_version
    preprocessor_path = model_dir / "preprocessor.pkl"

    if not preprocessor_path.exists():
        logger.error("Preprocessor not found: %s", preprocessor_path)
        logger.error("Run 'python run_training.py' first.")
        sys.exit(1)

    app.state.preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor from: %s", preprocessor_path)

    # ── Load XGBoost model ─────────────────────────────────────
    model_path = model_dir / "xgb_model.json"

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Run 'python run_training.py' first.")
        sys.exit(1)

    model = XGBClassifier()
    model.load_model(str(model_path))
    app.state.model = model
    logger.info(
        "Loaded XGBoost model from: %s (%d features)",
        model_path,
        model.n_features_in_,
    )

    logger.info("PharmaGuard API ready — model version: %s", model_version)

    yield  # ← Server runs here

    # ── Cleanup (nothing needed) ───────────────────────────────
    logger.info("PharmaGuard API shutting down")


def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application.

    - Registers the lifespan manager for model loading.
    - Adds CORS middleware for cross-origin access.
    - Includes all route handlers from ``src.api.routes``.

    Returns
    -------
    FastAPI
        Fully configured application instance.
    """
    app = FastAPI(
        title="PharmaGuard API",
        description=(
            "REST API for predicting serious Adverse Drug Reactions (ADRs) "
            "using an XGBoost classifier trained on FDA FAERS data.\n\n"
            "**Endpoints:**\n"
            "- `GET /` — Public dashboard with model metrics\n"
            "- `POST /predict` — Single ADR prediction\n"
            "- `POST /predict/batch` — Batch predictions\n"
            "- `GET /model/info` — Model metadata\n"
            "- `GET /model/history` — Historical training runs\n"
            "- `GET /health` — Health check\n"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS middleware ────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register routes ────────────────────────────────────────
    from src.api.routes import router
    app.include_router(router)

    # ── Mount static files (plots) ─────────────────────────────
    plots_dir = _PROJECT_ROOT / "metrics" / "runs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/plots", StaticFiles(directory=str(plots_dir)), name="plots")

    return app


# ── Module-level app instance (used by uvicorn) ───────────────────
app = create_app()
