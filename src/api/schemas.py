"""
PharmaGuard — API Request / Response Schemas
================================================
Pydantic models that define the shape of every request and response
handled by the PharmaGuard prediction API.

Using Pydantic v2 models gives us:
- Automatic request validation with detailed error messages
- OpenAPI (Swagger) documentation generation
- Type-safe serialisation / deserialisation
- Built-in JSON schema for client code generation
"""

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Request schemas
# ────────────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """
    Input schema for a single ADR severity prediction.

    All fields correspond to the features the XGBoost model was
    trained on.  The API preprocessor will one-hot encode categoricals
    and pass numericals through before inference.
    """

    sex: str = Field(
        ...,
        description="Patient sex code: M (male), F (female), or UNK (unknown)",
        examples=["F"],
    )
    age_group: str = Field(
        ...,
        description="Age group bucket from preprocessing (e.g. 0-17, 18-44, 45-64, 65-74, 75+, unknown)",
        examples=["45-64"],
    )
    route: str = Field(
        ...,
        description="Drug administration route (e.g. Oral, Intravenous, Topical, Unknown)",
        examples=["Oral"],
    )
    drug_name_normalized: str = Field(
        ...,
        description="Normalized drug name (uppercase, fuzzy-matched in Phase 2)",
        examples=["ASPIRIN"],
    )
    polypharmacy_count: int = Field(
        ...,
        ge=0,
        description="Number of concurrent drugs the patient is taking",
        examples=[3],
    )


class BatchPredictionRequest(BaseModel):
    """Batch wrapper: a list of prediction inputs for efficient bulk inference."""

    inputs: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of prediction inputs (max 1000 per batch)",
    )


# ────────────────────────────────────────────────────────────────────
# Response schemas
# ────────────────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """
    Output schema for a single prediction.

    Returns both the binary classification result and the raw
    probability so the consumer can apply their own threshold if needed.
    """

    is_serious: bool = Field(
        ...,
        description="Whether the ADR is predicted as serious (True) or not serious (False)",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model-predicted probability of the ADR being serious",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used for this prediction",
    )
    model_version: str = Field(
        ...,
        description="Version identifier of the serving model",
    )


class BatchPredictionResponse(BaseModel):
    """Batch wrapper: predictions for every input in the request."""

    predictions: list[PredictionResponse]
    count: int = Field(..., description="Number of predictions returned")


class HealthResponse(BaseModel):
    """Health check response confirming API and model readiness."""

    status: str = Field(..., description="Service status (healthy / degraded)")
    model_loaded: bool = Field(..., description="Whether the XGBoost model is loaded")
    timestamp: str = Field(..., description="Server timestamp (ISO 8601)")


class ModelInfoResponse(BaseModel):
    """Metadata about the currently served model."""

    model_type: str = Field(..., description="Model algorithm type")
    n_features: int = Field(..., description="Number of input features after preprocessing")
    threshold: float = Field(..., description="Current decision threshold")
    threshold_strategy: str = Field(..., description="Strategy used to tune the threshold")
    training_quarters: list = Field(..., description="FAERS quarters used for training")
    scale_pos_weight: float = Field(..., description="Class imbalance weight")
    model_version: str = Field(..., description="Model version identifier")
