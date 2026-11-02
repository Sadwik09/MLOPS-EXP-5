"""FastAPI application for Chronic Kidney Disease risk prediction.

Endpoints
---------
* ``GET  /``            service discovery
* ``GET  /health``      liveness probe
* ``GET  /ready``       readiness probe (503 until artifacts are loaded)
* ``GET  /model-info``  model card
* ``GET  /features``    dynamic form schema for the web UI
* ``POST /predict``     CKD probability for one patient
* ``GET  /metrics``     in-memory request counters
* ``GET  /ui``          single-page web interface
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.feature_spec import (
    DISCLAIMER,
    FEATURE_SPECS,
    MODEL_NAME,
    MODEL_VERSION,
    build_features_payload,
)
from app.metrics_store import MetricsStore
from app.model_loader import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, ModelLoader
from app.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PatientInput,
    PredictionResponse,
    ReadyResponse,
    RootResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ckd.api")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

DESCRIPTION = (
    "A machine-learning REST API that estimates the likelihood of Chronic Kidney "
    "Disease (CKD) from 24 routine clinical features. The service loads a "
    "scikit-learn pipeline (median/most-frequent imputation, scaling, one-hot "
    "encoding and a Random Forest) once at startup and exposes a JSON API plus "
    "a dynamic web UI.\n\n"
    "**Safety notice:** model output only, not a medical diagnosis."
)

loader = ModelLoader(
    model_path=os.getenv("CKD_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    metadata_path=os.getenv("CKD_METADATA_PATH", str(DEFAULT_METADATA_PATH)),
)
metrics = MetricsStore()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model artifacts once, before the app accepts traffic."""
    if loader.load():
        logger.info("CKD model is ready to serve predictions.")
    else:
        logger.error(
            "CKD model is NOT ready (%s). Run 'python train.py' to create artifacts. "
            "The service will stay up but /ready returns 503 and /predict is unavailable.",
            loader.load_error,
        )
    yield
    logger.info("Shutting down CKD Prediction API.")


app = FastAPI(
    title="CKD Prediction API",
    version=__version__,
    description=DESCRIPTION,
    summary="Chronic Kidney Disease risk prediction from routine clinical features",
    lifespan=lifespan,
    contact={"name": "CKD Prediction API", "url": "https://github.com/"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Health", "description": "Liveness and readiness probes."},
        {"name": "Model", "description": "Model card and feature schema."},
        {"name": "Prediction", "description": "CKD risk prediction."},
        {"name": "Observability", "description": "Runtime request metrics."},
        {"name": "UI", "description": "Browser interface."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Middleware & error handling
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def track_requests(request: Request, call_next: Any) -> Any:
    """Time every request and feed the in-memory metrics store."""
    started = time.perf_counter()
    is_prediction = request.method == "POST" and request.url.path.rstrip("/") == "/predict"
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics.record(elapsed_ms, 500, is_prediction)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    metrics.record(elapsed_ms, response.status_code, is_prediction)
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a UI-friendly 422 envelope with field-level messages."""
    labels = {spec["name"]: spec["label"] for spec in FEATURE_SPECS}
    errors: list[dict[str, Any]] = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", ()) if part != "body"]
        field = location[-1] if location else None
        error_type = error.get("type")
        message = error.get("msg", "Invalid value.")

        if error_type == "missing":
            message = "This field is required."
        elif error_type == "extra_forbidden":
            message = f"Unknown field '{field}'. Remove it from the request."
        elif error_type == "literal_error":
            message = f"Invalid value. {message}"

        errors.append(
            {
                "field": field,
                "label": labels.get(field, field),
                "message": message,
                "type": error_type,
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed. Please check the highlighted fields.",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Return JSON (never an HTML traceback) for unexpected failures."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again later.",
            "errors": [],
        },
    )


# --------------------------------------------------------------------------- #
# Discovery & probes
# --------------------------------------------------------------------------- #
@app.get("/", response_model=RootResponse, tags=["Health"], summary="Service discovery")
def root() -> RootResponse:
    """Return API identity and links to the interactive documentation and UI."""
    return RootResponse(
        name="CKD Prediction API",
        version=__version__,
        description="Chronic Kidney Disease risk prediction from routine clinical features.",
        docs="/docs",
        ui="/ui",
        health="/health",
        openapi="/openapi.json",
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Liveness probe")
def health() -> HealthResponse:
    """Report that the process is alive (does not check model readiness)."""
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"description": "Model artifacts are not loaded"}},
    tags=["Health"],
    summary="Readiness probe",
)
def ready() -> Any:
    """Return 200 only when both the pipeline and metadata are loaded."""
    if loader.ready:
        return ReadyResponse(ready=True)
    return JSONResponse(
        status_code=503,
        content={"ready": False, "reason": "Model artifacts are not loaded. Run 'python train.py'."},
    )


# --------------------------------------------------------------------------- #
# Model metadata & features
# --------------------------------------------------------------------------- #
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"], summary="Model card")
def model_info() -> ModelInfoResponse:
    """Return model identity, feature schema, evaluation metrics and training date."""
    if not loader.ready:
        raise HTTPException(status_code=503, detail="Model metadata is not loaded. Run 'python train.py'.")
    metadata = loader.metadata
    return ModelInfoResponse(
        model_name=metadata.get("model_name", MODEL_NAME),
        model_version=metadata.get("model_version", MODEL_VERSION),
        algorithm=metadata.get("algorithm", "RandomForestClassifier"),
        n_features=int(metadata.get("n_features", len(FEATURE_SPECS))),
        feature_names=list(metadata.get("feature_names", [])),
        features=list(metadata.get("features", [])),
        evaluation=dict(metadata.get("evaluation", {})),
        training_date=metadata.get("training_date", ""),
        dataset_size=int(metadata.get("dataset_size", 0)),
        disclaimer=metadata.get("disclaimer", DISCLAIMER),
    )


@app.get("/features", tags=["Model"], summary="Dynamic form schema for the web UI")
def features() -> dict[str, Any]:
    """Return field definitions, sections, ranges and demo samples.

    The UI builds every input from this response, so adding a feature to
    ``app/feature_spec.py`` and retraining automatically updates the form.
    """
    numeric_ranges = loader.metadata.get("numeric_ranges", {}) if loader.ready else {}
    payload = build_features_payload(numeric_ranges)
    payload["model_version"] = loader.metadata.get("model_version", MODEL_VERSION) if loader.ready else None
    return payload


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"description": "Invalid or missing input fields"},
        503: {"description": "Model is not loaded"},
    },
    tags=["Prediction"],
    summary="Predict CKD risk for one patient",
)
def predict(patient: PatientInput) -> PredictionResponse:
    """Run the trained pipeline and return CKD / Not CKD with probabilities.

    Missing optional fields are imputed by the pipeline using training
    statistics. The response always includes the not-a-diagnosis disclaimer.
    """
    if not loader.ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Artifacts are missing; run 'python train.py'.",
        )

    try:
        label, probability_ckd, probability_not_ckd = loader.predict(patient.model_dump())
    except Exception as exc:  # noqa: BLE001 - converted to a clean API error
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    metadata = loader.metadata
    logger.info(
        "Prediction served: %s (p_ckd=%.4f, model_version=%s)",
        label,
        probability_ckd,
        metadata.get("model_version"),
    )
    return PredictionResponse(
        prediction=label,  # type: ignore[arg-type]
        probability_ckd=round(probability_ckd, 6),
        probability_not_ckd=round(probability_not_ckd, 6),
        model_version=str(metadata.get("model_version", MODEL_VERSION)),
        disclaimer=DISCLAIMER,
    )


# --------------------------------------------------------------------------- #
# Metrics & UI
# --------------------------------------------------------------------------- #
@app.get("/metrics", response_model=MetricsResponse, tags=["Observability"], summary="Request metrics")
def get_metrics() -> MetricsResponse:
    """Return request counters and average latency since process start."""
    return MetricsResponse(**metrics.snapshot())


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the single-page web interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI files are missing.")
    return FileResponse(index_path, media_type="text/html")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
