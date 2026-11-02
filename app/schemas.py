"""Pydantic v2 request/response models for the CKD Prediction API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.feature_spec import DISCLAIMER, SAMPLES


class PatientInput(BaseModel):
    """A single patient record accepted by ``POST /predict``.

    Only ``age``, ``bp``, ``sg``, ``al``, ``hemo`` and ``sc`` are required by the
    API contract. Every other field may be omitted or set to ``null``; the
    pipeline imputes missing values using statistics learned during training.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [SAMPLES["ckd"], SAMPLES["not_ckd"]]},
    )

    #: Basic vitals -----------------------------------------------------------
    age: float = Field(..., ge=0, le=120, description="Age in years", examples=[55])
    bp: float = Field(..., ge=20, le=250, description="Diastolic blood pressure (mmHg)", examples=[80])

    #: Urine tests ------------------------------------------------------------
    sg: Literal[1.005, 1.01, 1.015, 1.02, 1.025] = Field(
        ..., description="Urine specific gravity (1.005, 1.010, 1.015, 1.020 or 1.025)"
    )
    al: int = Field(..., ge=0, le=5, description="Albumin in urine, 0-5 scale", examples=[1])
    su: Optional[int] = Field(None, ge=0, le=5, description="Sugar in urine, 0-5 scale")
    rbc: Optional[Literal["normal", "abnormal"]] = Field(None, description="Red blood cells in urine")
    pc: Optional[Literal["normal", "abnormal"]] = Field(None, description="Pus cells in urine")
    pcc: Optional[Literal["present", "notpresent"]] = Field(None, description="Pus cell clumps")
    ba: Optional[Literal["present", "notpresent"]] = Field(None, description="Bacteria in urine")

    #: Blood tests ------------------------------------------------------------
    bgr: Optional[float] = Field(None, ge=20, le=800, description="Random blood glucose (mg/dL)")
    bu: Optional[float] = Field(None, ge=0, le=500, description="Blood urea (mg/dL)")
    sc: float = Field(..., ge=0, le=100, description="Serum creatinine (mg/dL)", examples=[1.2])
    sod: Optional[float] = Field(None, ge=1, le=200, description="Sodium (mEq/L)")
    pot: Optional[float] = Field(None, ge=1, le=60, description="Potassium (mEq/L)")
    hemo: float = Field(..., ge=3, le=20, description="Hemoglobin (g/dL)", examples=[13.5])
    pcv: Optional[float] = Field(None, ge=5, le=60, description="Packed cell volume (%)")
    wc: Optional[float] = Field(None, ge=1000, le=40000, description="White blood cell count (cells/cmm)")
    rc: Optional[float] = Field(None, ge=1, le=10, description="Red blood cell count (millions/cmm)")

    #: Medical history & symptoms --------------------------------------------
    htn: Optional[Literal["yes", "no"]] = Field(None, description="Hypertension")
    dm: Optional[Literal["yes", "no"]] = Field(None, description="Diabetes mellitus")
    cad: Optional[Literal["yes", "no"]] = Field(None, description="Coronary artery disease")
    appet: Optional[Literal["good", "poor"]] = Field(None, description="Appetite")
    pe: Optional[Literal["yes", "no"]] = Field(None, description="Pedal edema")
    ane: Optional[Literal["yes", "no"]] = Field(None, description="Anemia")


class PredictionResponse(BaseModel):
    """Result of a single CKD risk prediction."""

    prediction: Literal["CKD", "Not CKD"] = Field(..., description="Human-readable class label")
    probability_ckd: float = Field(..., ge=0, le=1, description="Model probability of CKD")
    probability_not_ckd: float = Field(..., ge=0, le=1, description="Model probability of Not CKD")
    model_version: str = Field(..., description="Version of the model that served the request")
    disclaimer: str = Field(DISCLAIMER, description="Safety disclaimer")


class RootResponse(BaseModel):
    """Service discovery information."""

    name: str
    version: str
    description: str
    docs: str
    ui: str
    health: str
    openapi: str


class HealthResponse(BaseModel):
    """Liveness response."""

    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    """Readiness response."""

    ready: bool


class MetricsResponse(BaseModel):
    """Runtime request counters."""

    total_requests: int
    successful_predictions: int
    failed_requests: int
    average_response_time_ms: float
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model card summary served from the training metadata."""

    model_name: str
    model_version: str
    algorithm: str
    n_features: int
    feature_names: list[str]
    features: list[dict[str, Any]]
    evaluation: dict[str, Any]
    training_date: str
    dataset_size: int
    disclaimer: str


class ErrorDetail(BaseModel):
    """A single field-level validation error."""

    field: Optional[str] = None
    label: Optional[str] = None
    message: str
    type: Optional[str] = None


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by validation and catch-all handlers."""

    detail: str
    errors: list[ErrorDetail] = Field(default_factory=list)
