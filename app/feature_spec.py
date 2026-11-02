"""Single source of truth for the CKD model features.

The same definitions are used by the training script (to build metadata), the
FastAPI schemas (to validate input) and the web UI (to build the form), which
keeps the three layers from drifting apart.

Feature order in :data:`FEATURE_SPECS` is the exact column order used to fit
the scikit-learn pipeline.
"""

from __future__ import annotations

from typing import Any

MODEL_VERSION = "1.0.0"
MODEL_NAME = "CKD Random Forest Classifier"
TARGET_NAME = "classification"
RANDOM_STATE = 42
DISCLAIMER = "Model output only, not a medical diagnosis."

#: Canonical urine specific-gravity levels in the UCI dataset. Stored as floats
#: because JSON and pandas do not preserve trailing zeros (1.010 -> 1.01).
SG_LEVELS: tuple[float, ...] = (1.005, 1.01, 1.015, 1.02, 1.025)


def _options(values: list[Any], labels: list[str] | None = None) -> list[dict[str, Any]]:
    """Return ``{value, label}`` option dictionaries for a discrete field."""
    if labels is None:
        labels = [str(value).capitalize() for value in values]
    return [{"value": value, "label": label} for value, label in zip(values, labels)]


CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"],
}

_CATEGORICAL_LABELS: dict[str, list[str]] = {
    "rbc": ["Normal", "Abnormal"],
    "pc": ["Normal", "Abnormal"],
    "pcc": ["Present", "Not present"],
    "ba": ["Present", "Not present"],
    "htn": ["Yes", "No"],
    "dm": ["Yes", "No"],
    "cad": ["Yes", "No"],
    "appet": ["Good", "Poor"],
    "pe": ["Yes", "No"],
    "ane": ["Yes", "No"],
}

#: Ordered field definitions. ``min``/``max`` are API acceptance bounds;
#: observed training ranges are merged in from ``model_metadata.json`` at
#: runtime so the UI can show realistic hints.
FEATURE_SPECS: list[dict[str, Any]] = [
    {
        "name": "age",
        "label": "Age",
        "type": "numeric",
        "dtype": "float",
        "section": "vitals",
        "required": True,
        "unit": "years",
        "description": "Patient age in years.",
        "min": 0,
        "max": 120,
        "step": 1,
    },
    {
        "name": "bp",
        "label": "Blood Pressure",
        "type": "numeric",
        "dtype": "float",
        "section": "vitals",
        "required": True,
        "unit": "mmHg",
        "description": "Diastolic blood pressure reading.",
        "min": 20,
        "max": 250,
        "step": 1,
    },
    {
        "name": "sg",
        "label": "Urine Specific Gravity",
        "type": "numeric",
        "dtype": "float",
        "section": "urine",
        "required": True,
        "unit": "",
        "description": "Urine concentration, one of five measured levels.",
        "min": 1.005,
        "max": 1.025,
        "step": 0.005,
        "options": _options(list(SG_LEVELS), ["1.005", "1.010", "1.015", "1.020", "1.025"]),
    },
    {
        "name": "al",
        "label": "Albumin in Urine",
        "type": "numeric",
        "dtype": "integer",
        "section": "urine",
        "required": True,
        "unit": "0–5 scale",
        "description": "Albumin detected in urine (0 = none, 5 = heavy).",
        "min": 0,
        "max": 5,
        "step": 1,
    },
    {
        "name": "su",
        "label": "Sugar in Urine",
        "type": "numeric",
        "dtype": "integer",
        "section": "urine",
        "required": False,
        "unit": "0–5 scale",
        "description": "Glucose detected in urine (0 = none, 5 = heavy).",
        "min": 0,
        "max": 5,
        "step": 1,
    },
    {
        "name": "rbc",
        "label": "Red Blood Cells in Urine",
        "type": "categorical",
        "dtype": "string",
        "section": "urine",
        "required": False,
        "unit": "",
        "description": "Red blood cells present in the urine sample.",
        "options": _options(CATEGORICAL_OPTIONS["rbc"], _CATEGORICAL_LABELS["rbc"]),
    },
    {
        "name": "pc",
        "label": "Pus Cells in Urine",
        "type": "categorical",
        "dtype": "string",
        "section": "urine",
        "required": False,
        "unit": "",
        "description": "Pus (white blood) cells present in urine.",
        "options": _options(CATEGORICAL_OPTIONS["pc"], _CATEGORICAL_LABELS["pc"]),
    },
    {
        "name": "pcc",
        "label": "Pus Cell Clumps",
        "type": "categorical",
        "dtype": "string",
        "section": "urine",
        "required": False,
        "unit": "",
        "description": "Clumps of pus cells seen in urine.",
        "options": _options(CATEGORICAL_OPTIONS["pcc"], _CATEGORICAL_LABELS["pcc"]),
    },
    {
        "name": "ba",
        "label": "Bacteria in Urine",
        "type": "categorical",
        "dtype": "string",
        "section": "urine",
        "required": False,
        "unit": "",
        "description": "Bacteria detected in the urine sample.",
        "options": _options(CATEGORICAL_OPTIONS["ba"], _CATEGORICAL_LABELS["ba"]),
    },
    {
        "name": "bgr",
        "label": "Blood Glucose Random",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "mg/dL",
        "description": "Random blood glucose measurement.",
        "min": 20,
        "max": 800,
        "step": 1,
    },
    {
        "name": "bu",
        "label": "Blood Urea",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "mg/dL",
        "description": "Blood urea nitrogen level.",
        "min": 0,
        "max": 500,
        "step": 1,
    },
    {
        "name": "sc",
        "label": "Serum Creatinine",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": True,
        "unit": "mg/dL",
        "description": "Serum creatinine, a key kidney-function marker.",
        "min": 0,
        "max": 100,
        "step": 0.1,
    },
    {
        "name": "sod",
        "label": "Sodium",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "mEq/L",
        "description": "Serum sodium concentration.",
        "min": 1,
        "max": 200,
        "step": 0.1,
    },
    {
        "name": "pot",
        "label": "Potassium",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "mEq/L",
        "description": "Serum potassium concentration.",
        "min": 1,
        "max": 60,
        "step": 0.1,
    },
    {
        "name": "hemo",
        "label": "Hemoglobin",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": True,
        "unit": "g/dL",
        "description": "Hemoglobin concentration in blood.",
        "min": 3,
        "max": 20,
        "step": 0.1,
    },
    {
        "name": "pcv",
        "label": "Packed Cell Volume",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "%",
        "description": "Packed cell volume (hematocrit).",
        "min": 5,
        "max": 60,
        "step": 0.1,
    },
    {
        "name": "wc",
        "label": "White Blood Cell Count",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "cells/cmm",
        "description": "White blood cell count.",
        "min": 1000,
        "max": 40000,
        "step": 100,
    },
    {
        "name": "rc",
        "label": "Red Blood Cell Count",
        "type": "numeric",
        "dtype": "float",
        "section": "blood",
        "required": False,
        "unit": "millions/cmm",
        "description": "Red blood cell count.",
        "min": 1,
        "max": 10,
        "step": 0.1,
    },
    {
        "name": "htn",
        "label": "Hypertension",
        "type": "categorical",
        "dtype": "string",
        "section": "history",
        "required": False,
        "unit": "",
        "description": "Has the patient been diagnosed with hypertension?",
        "options": _options(CATEGORICAL_OPTIONS["htn"], _CATEGORICAL_LABELS["htn"]),
    },
    {
        "name": "dm",
        "label": "Diabetes Mellitus",
        "type": "categorical",
        "dtype": "string",
        "section": "history",
        "required": False,
        "unit": "",
        "description": "Has the patient been diagnosed with diabetes?",
        "options": _options(CATEGORICAL_OPTIONS["dm"], _CATEGORICAL_LABELS["dm"]),
    },
    {
        "name": "cad",
        "label": "Coronary Artery Disease",
        "type": "categorical",
        "dtype": "string",
        "section": "history",
        "required": False,
        "unit": "",
        "description": "History of coronary artery disease.",
        "options": _options(CATEGORICAL_OPTIONS["cad"], _CATEGORICAL_LABELS["cad"]),
    },
    {
        "name": "appet",
        "label": "Appetite",
        "type": "categorical",
        "dtype": "string",
        "section": "history",
        "required": False,
        "unit": "",
        "description": "Patient's appetite level.",
        "options": _options(CATEGORICAL_OPTIONS["appet"], _CATEGORICAL_LABELS["appet"]),
    },
    {
        "name": "pe",
        "label": "Pedal Edema",
        "type": "categorical",
        "dtype": "string",
        "section": "history",
        "required": False,
        "unit": "",
        "description": "Fluid swelling in the feet or ankles.",
        "options": _options(CATEGORICAL_OPTIONS["pe"], _CATEGORICAL_LABELS["pe"]),
    },
    {
        "name": "ane",
        "label": "Anemia",
        "type": "categorical",
        "dtype": "string",
        "section": "history",
        "required": False,
        "unit": "",
        "description": "Has the patient been diagnosed with anemia?",
        "options": _options(CATEGORICAL_OPTIONS["ane"], _CATEGORICAL_LABELS["ane"]),
    },
]

SECTIONS: list[dict[str, str]] = [
    {
        "id": "vitals",
        "title": "Basic vitals",
        "description": "Core demographic and blood-pressure measurements.",
    },
    {
        "id": "urine",
        "title": "Urine tests",
        "description": "Routine urinalysis results.",
    },
    {
        "id": "blood",
        "title": "Blood tests",
        "description": "Laboratory blood panel values.",
    },
    {
        "id": "history",
        "title": "Medical history & symptoms",
        "description": "Known conditions and clinical signs.",
    },
]

FEATURE_NAMES: list[str] = [spec["name"] for spec in FEATURE_SPECS]
NUMERIC_FEATURES: list[str] = [s["name"] for s in FEATURE_SPECS if s["type"] == "numeric"]
CATEGORICAL_FEATURES: list[str] = [s["name"] for s in FEATURE_SPECS if s["type"] == "categorical"]
REQUIRED_FEATURES: list[str] = [s["name"] for s in FEATURE_SPECS if s["required"]]

#: Curated, fully-populated demo patients used by the UI sample buttons.
SAMPLES: dict[str, dict[str, Any]] = {
    "ckd": {
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
    },
    "not_ckd": {
        "age": 35,
        "bp": 70,
        "sg": 1.02,
        "al": 0,
        "su": 0,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 95,
        "bu": 25,
        "sc": 0.9,
        "sod": 140,
        "pot": 4.2,
        "hemo": 15.2,
        "pcv": 45,
        "wc": 7200,
        "rc": 5.1,
        "htn": "no",
        "dm": "no",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    },
}


def build_features_payload(numeric_ranges: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Build the response body for ``GET /features``.

    Args:
        numeric_ranges: Optional ``{feature: {"min": ..., "max": ...}}`` map
            taken from the saved metadata. Missing entries are simply omitted.

    Returns:
        A JSON-ready dictionary containing flat features, grouped sections and
        demo samples.
    """
    ranges = numeric_ranges or {}
    features: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        item = dict(spec)
        if spec["type"] == "numeric" and spec["name"] in ranges:
            item["observed_min"] = ranges[spec["name"]]["min"]
            item["observed_max"] = ranges[spec["name"]]["max"]
        features.append(item)

    by_name = {feature["name"]: feature for feature in features}
    sections = [
        {
            "id": section["id"],
            "title": section["title"],
            "description": section["description"],
            "fields": [by_name[spec["name"]] for spec in FEATURE_SPECS if spec["section"] == section["id"]],
        }
        for section in SECTIONS
    ]
    return {
        "sections": sections,
        "features": features,
        "required": REQUIRED_FEATURES,
        "samples": SAMPLES,
        "disclaimer": DISCLAIMER,
    }
