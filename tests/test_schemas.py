"""
PharmaGuard — Schema Validation Tests
==========================================
Tests that Pydantic request/response models accept valid input and
reject invalid input with proper validation errors.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


# ────────────────────────────────────────────────────────────────────
# PredictionRequest
# ────────────────────────────────────────────────────────────────────


class TestPredictionRequest:
    """Test the PredictionRequest Pydantic model."""

    def test_valid_request(self):
        """A well-formed request should parse without errors."""
        req = PredictionRequest(
            sex="F",
            age_group="45-64",
            route="Oral",
            drug_name_normalized="ASPIRIN",
            polypharmacy_count=3,
        )
        assert req.sex == "F"
        assert req.age_group == "45-64"
        assert req.route == "Oral"
        assert req.drug_name_normalized == "ASPIRIN"
        assert req.polypharmacy_count == 3

    def test_missing_required_field(self):
        """Omitting a required field should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                sex="M",
                age_group="18-44",
                # route is missing
                drug_name_normalized="IBUPROFEN",
                polypharmacy_count=1,
            )
        assert "route" in str(exc_info.value)

    def test_negative_polypharmacy_count(self):
        """polypharmacy_count must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                sex="M",
                age_group="65-74",
                route="Intravenous",
                drug_name_normalized="WARFARIN",
                polypharmacy_count=-1,
            )
        assert "polypharmacy_count" in str(exc_info.value)

    def test_zero_polypharmacy_is_valid(self):
        """polypharmacy_count=0 (monotherapy) should be accepted."""
        req = PredictionRequest(
            sex="F",
            age_group="0-17",
            route="Topical",
            drug_name_normalized="HYDROCORTISONE",
            polypharmacy_count=0,
        )
        assert req.polypharmacy_count == 0

    def test_model_dump_returns_dict(self):
        """model_dump() should return a serializable dictionary."""
        req = PredictionRequest(
            sex="M",
            age_group="75+",
            route="Oral",
            drug_name_normalized="METFORMIN",
            polypharmacy_count=5,
        )
        data = req.model_dump()
        assert isinstance(data, dict)
        assert data["sex"] == "M"
        assert data["polypharmacy_count"] == 5


# ────────────────────────────────────────────────────────────────────
# BatchPredictionRequest
# ────────────────────────────────────────────────────────────────────


class TestBatchPredictionRequest:
    """Test the BatchPredictionRequest Pydantic model."""

    def test_valid_batch(self):
        """A batch with 2 inputs should be accepted."""
        batch = BatchPredictionRequest(
            inputs=[
                PredictionRequest(
                    sex="F", age_group="45-64", route="Oral",
                    drug_name_normalized="ASPIRIN", polypharmacy_count=3,
                ),
                PredictionRequest(
                    sex="M", age_group="18-44", route="Intravenous",
                    drug_name_normalized="IBUPROFEN", polypharmacy_count=1,
                ),
            ]
        )
        assert len(batch.inputs) == 2

    def test_empty_batch_rejected(self):
        """An empty inputs list should be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(inputs=[])


# ────────────────────────────────────────────────────────────────────
# PredictionResponse
# ────────────────────────────────────────────────────────────────────


class TestPredictionResponse:
    """Test the PredictionResponse Pydantic model."""

    def test_valid_response(self):
        """A well-formed response should parse correctly."""
        resp = PredictionResponse(
            is_serious=True,
            probability=0.85,
            threshold=0.24,
            model_version="v2",
        )
        assert resp.is_serious is True
        assert resp.probability == 0.85

    def test_probability_out_of_range(self):
        """probability must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                is_serious=False,
                probability=1.5,  # Invalid
                threshold=0.5,
                model_version="v1",
            )


# ────────────────────────────────────────────────────────────────────
# HealthResponse
# ────────────────────────────────────────────────────────────────────


class TestHealthResponse:
    """Test the HealthResponse schema."""

    def test_valid_health_response(self):
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            timestamp="2026-05-31T15:00:00",
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True


# ────────────────────────────────────────────────────────────────────
# ModelInfoResponse
# ────────────────────────────────────────────────────────────────────


class TestModelInfoResponse:
    """Test the ModelInfoResponse schema."""

    def test_valid_model_info(self):
        resp = ModelInfoResponse(
            model_type="XGBoost (XGBClassifier)",
            n_features=396,
            threshold=0.24,
            threshold_strategy="f1",
            training_quarters=["2026q1"],
            scale_pos_weight=0.38,
            model_version="v2",
        )
        assert resp.n_features == 396
        assert resp.threshold == 0.24
