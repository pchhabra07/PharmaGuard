"""
PharmaGuard — API Route Tests
==================================
Tests all HTTP endpoints using an async client connected to a
FastAPI test app with mocked model artifacts.

Each test verifies:
- Correct HTTP status code
- Response JSON structure matches the expected schema
- Edge cases (bad input, missing fields) return proper errors
"""

import pytest


# ────────────────────────────────────────────────────────────────────
# Dashboard (GET /)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_returns_html(client):
    """GET / should return an HTML dashboard page."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "PharmaGuard" in resp.text


# ────────────────────────────────────────────────────────────────────
# Health check (GET /health)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_healthy(client):
    """GET /health should return status=healthy when model is loaded."""
    resp = await client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data


# ────────────────────────────────────────────────────────────────────
# Model info (GET /model/info)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_info(client):
    """GET /model/info should return model metadata."""
    resp = await client.get("/model/info")
    assert resp.status_code == 200

    data = resp.json()
    assert data["model_type"] == "XGBoost (XGBClassifier)"
    assert data["n_features"] == 396
    assert data["threshold"] == 0.24
    assert data["threshold_strategy"] == "f1"
    assert data["model_version"] == "v2"
    assert "training_quarters" in data


# ────────────────────────────────────────────────────────────────────
# Single prediction (POST /predict)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_serious(client):
    """
    POST /predict with valid input should return a prediction.

    The mock model returns probability=0.85, which is above the
    threshold of 0.24, so is_serious should be True.
    """
    payload = {
        "sex": "F",
        "age_group": "45-64",
        "route": "Oral",
        "drug_name_normalized": "ASPIRIN",
        "polypharmacy_count": 3,
    }
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["is_serious"] is True
    assert data["probability"] == 0.85
    assert data["threshold"] == 0.24
    assert data["model_version"] == "v2"


@pytest.mark.asyncio
async def test_predict_missing_field(client):
    """POST /predict with a missing required field should return 422."""
    payload = {
        "sex": "M",
        "age_group": "18-44",
        # "route" is missing
        "drug_name_normalized": "IBUPROFEN",
        "polypharmacy_count": 1,
    }
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_predict_invalid_polypharmacy(client):
    """POST /predict with negative polypharmacy_count should return 422."""
    payload = {
        "sex": "F",
        "age_group": "65-74",
        "route": "Intravenous",
        "drug_name_normalized": "WARFARIN",
        "polypharmacy_count": -5,
    }
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422


# ────────────────────────────────────────────────────────────────────
# Batch prediction (POST /predict/batch)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_batch(client):
    """POST /predict/batch should return predictions for all inputs."""
    payload = {
        "inputs": [
            {
                "sex": "F",
                "age_group": "45-64",
                "route": "Oral",
                "drug_name_normalized": "ASPIRIN",
                "polypharmacy_count": 3,
            },
            {
                "sex": "M",
                "age_group": "18-44",
                "route": "Intravenous",
                "drug_name_normalized": "IBUPROFEN",
                "polypharmacy_count": 1,
            },
        ]
    }
    resp = await client.post("/predict/batch", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    # Each prediction should have the expected fields
    for pred in data["predictions"]:
        assert "is_serious" in pred
        assert "probability" in pred
        assert "threshold" in pred
        assert "model_version" in pred


@pytest.mark.asyncio
async def test_predict_batch_empty(client):
    """POST /predict/batch with empty inputs should return 422."""
    resp = await client.post("/predict/batch", json={"inputs": []})
    assert resp.status_code == 422


# ────────────────────────────────────────────────────────────────────
# Model charts (GET /model/charts)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_charts(client):
    """GET /model/charts should return chart-ready data."""
    resp = await client.get("/model/charts")
    assert resp.status_code == 200

    data = resp.json()
    assert "runs_count" in data
    # With no metrics file, latest may be None
    assert "latest" in data
    assert "history_trend" in data


# ────────────────────────────────────────────────────────────────────
# Swagger docs (GET /docs)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_swagger_docs(client):
    """GET /docs should return the Swagger UI HTML page."""
    resp = await client.get("/docs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
