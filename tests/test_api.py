"""End-to-end API tests using FastAPI's TestClient.

Run with::

    pytest -q

The tests expect ``artifacts/ckd_pipeline.joblib`` and
``artifacts/model_metadata.json`` to exist (run ``python train.py`` first).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app

DISCLAIMER = "Model output only, not a medical diagnosis."


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Start the app once for the whole module (runs the lifespan loader)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload() -> dict[str, Any]:
    """A clinically plausible CKD patient with every field populated."""
    return {
        "age": 60,
        "bp": 80,
        "sg": 1.01,
        "al": 3,
        "su": 4,
        "rbc": "abnormal",
        "pc": "abnormal",
        "pcc": "present",
        "ba": "present",
        "bgr": 423,
        "bu": 53,
        "sc": 1.8,
        "sod": 137,
        "pot": 4.5,
        "hemo": 9.6,
        "pcv": 31,
        "wc": 7500,
        "rc": 3.9,
        "htn": "yes",
        "dm": "yes",
        "cad": "no",
        "appet": "poor",
        "pe": "no",
        "ane": "yes",
    }


@pytest.fixture()
def minimal_payload() -> dict[str, Any]:
    """Only the six required fields."""
    return {"age": 55, "bp": 80, "sg": 1.02, "al": 0, "hemo": 14.2, "sc": 1.1}


# --------------------------------------------------------------------------- #
# Discovery & probes
# --------------------------------------------------------------------------- #
def test_root_returns_service_info(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "CKD Prediction API"
    assert body["version"] == "1.0.0"
    assert body["docs"] == "/docs"
    assert body["ui"] == "/ui"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "CKD Random Forest Classifier"
    assert body["model_version"] == "1.0.0"
    assert body["n_features"] == 24
    assert len(body["feature_names"]) == 24
    assert body["dataset_size"] == 400
    assert 0.0 <= body["evaluation"]["accuracy"] <= 1.0
    assert "confusion_matrix" in body["evaluation"]
    assert body["training_date"]
    assert body["disclaimer"] == DISCLAIMER


def test_features_schema(client: TestClient) -> None:
    response = client.get("/features")
    assert response.status_code == 200
    body = response.json()
    assert len(body["features"]) == 24
    assert len(body["sections"]) == 4
    assert set(body["required"]) == {"age", "bp", "sg", "al", "hemo", "sc"}
    assert set(body["samples"]) == {"ckd", "not_ckd"}

    names = {feature["name"] for feature in body["features"]}
    assert "wc" in names and "rc" in names and "classification" not in names

    by_name = {feature["name"]: feature for feature in body["features"]}
    assert by_name["hemo"]["unit"] == "g/dL"
    assert by_name["age"]["min"] == 0 and by_name["age"]["max"] == 120
    assert by_name["bgr"]["observed_min"] >= 0
    assert {opt["value"] for opt in by_name["rbc"]["options"]} == {"normal", "abnormal"}
    assert {opt["value"] for opt in by_name["sg"]["options"]} == {1.005, 1.01, 1.015, 1.02, 1.025}


def test_ui_is_served(client: TestClient) -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NephroCheck" in response.text
    assert "/static/app.js" in response.text

    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]


# --------------------------------------------------------------------------- #
# Prediction — happy paths
# --------------------------------------------------------------------------- #
def test_predict_with_all_fields(client: TestClient, valid_payload: dict[str, Any]) -> None:
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"CKD", "Not CKD"}
    assert body["prediction"] == "CKD"  # this demo patient is a clear CKD case
    assert 0.0 <= body["probability_ckd"] <= 1.0
    assert 0.0 <= body["probability_not_ckd"] <= 1.0
    assert body["probability_ckd"] + body["probability_not_ckd"] == pytest.approx(1.0, abs=1e-6)
    assert body["model_version"] == "1.0.0"
    assert body["disclaimer"] == DISCLAIMER


def test_predict_with_only_required_fields(client: TestClient, minimal_payload: dict[str, Any]) -> None:
    response = client.post("/predict", json=minimal_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"CKD", "Not CKD"}
    assert body["probability_ckd"] + body["probability_not_ckd"] == pytest.approx(1.0, abs=1e-6)


def test_predict_null_optional_fields_are_imputed(client: TestClient, minimal_payload: dict[str, Any]) -> None:
    payload = {**minimal_payload, "rbc": None, "pc": None, "bgr": None, "htn": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in {"CKD", "Not CKD"}


def test_predict_healthy_sample(client: TestClient) -> None:
    healthy = {
        "age": 35, "bp": 70, "sg": 1.02, "al": 0, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 95, "bu": 25, "sc": 0.9, "sod": 140, "pot": 4.2,
        "hemo": 15.2, "pcv": 45, "wc": 7200, "rc": 5.1,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
    }
    response = client.post("/predict", json=healthy)
    assert response.status_code == 200
    assert response.json()["prediction"] == "Not CKD"


# --------------------------------------------------------------------------- #
# Prediction — validation failures
# --------------------------------------------------------------------------- #
def test_missing_required_field_returns_422(client: TestClient, valid_payload: dict[str, Any]) -> None:
    payload = dict(valid_payload)
    del payload["age"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], str)
    assert any(error["field"] == "age" for error in body["errors"])
    assert any("required" in error["message"].lower() for error in body["errors"])


def test_wrong_data_type_returns_422(client: TestClient, valid_payload: dict[str, Any]) -> None:
    payload = {**valid_payload, "age": "abc"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert any(error["field"] == "age" for error in body["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", -5),
        ("age", 200),
        ("hemo", 500),
        ("bp", 10),
        ("sc", -1),
        ("al", 9),
        ("bgr", 5),
        ("wc", 1),
        ("pot", 100),
    ],
)
def test_out_of_range_values_return_422(
    client: TestClient, valid_payload: dict[str, Any], field: str, value: float
) -> None:
    payload = {**valid_payload, field: value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert any(error["field"] == field for error in body["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rbc", "maybe"),
        ("pc", "present"),
        ("htn", "true"),
        ("appet", "medium"),
        ("sg", 1.0),
        ("dm", "notckd"),
    ],
)
def test_invalid_categorical_value_returns_422(
    client: TestClient, valid_payload: dict[str, Any], field: str, value: Any
) -> None:
    payload = {**valid_payload, field: value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert any(error["field"] == field for error in body["errors"])


def test_unknown_field_returns_422(client: TestClient, valid_payload: dict[str, Any]) -> None:
    payload = {**valid_payload, "shoe_size": 42}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.status_code == 422
    body = response.json()
    assert any(error["field"] == "shoe_size" for error in body["errors"])


def test_empty_body_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422
    fields = {error["field"] for error in response.json()["errors"]}
    assert {"age", "bp", "sg", "al", "hemo", "sc"}.issubset(fields)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_counters_track_predictions_and_failures(client: TestClient, valid_payload: dict[str, Any]) -> None:
    before = client.get("/metrics").json()
    assert set(before) == {
        "total_requests",
        "successful_predictions",
        "failed_requests",
        "average_response_time_ms",
        "uptime_seconds",
    }

    assert client.post("/predict", json=valid_payload).status_code == 200
    middle = client.get("/metrics").json()
    assert middle["successful_predictions"] == before["successful_predictions"] + 1
    assert middle["total_requests"] >= before["total_requests"] + 1
    assert middle["average_response_time_ms"] >= 0

    bad = client.post("/predict", json={**valid_payload, "age": "not-a-number"})
    assert bad.status_code == 422
    after = client.get("/metrics").json()
    assert after["failed_requests"] == middle["failed_requests"] + 1
    assert after["successful_predictions"] == middle["successful_predictions"]
    assert after["total_requests"] > middle["total_requests"]
    assert after["uptime_seconds"] >= 0


def test_metrics_has_numeric_types(client: TestClient) -> None:
    body = client.get("/metrics").json()
    assert isinstance(body["total_requests"], int)
    assert isinstance(body["successful_predictions"], int)
    assert isinstance(body["failed_requests"], int)
    assert isinstance(body["average_response_time_ms"], (int, float))
    assert isinstance(body["uptime_seconds"], (int, float))


def test_openapi_schema(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "/features" in schema["paths"]
    assert "PatientInput" in schema["components"]["schemas"]
