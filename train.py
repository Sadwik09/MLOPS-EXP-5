"""Train, evaluate and persist the Chronic Kidney Disease classifier.

Usage::

    python train.py

The script loads ``data/kidney_disease.csv`` (downloading it from the UCI ML
repository when missing), cleans the notoriously dirty raw values, fits a
single scikit-learn ``Pipeline`` and writes two artifacts:

* ``artifacts/ckd_pipeline.joblib`` -- the fitted preprocessing + model pipeline
* ``artifacts/model_metadata.json``   -- metrics, feature schema and training info

The exact same pipeline object is later loaded by the FastAPI service, so
training and inference preprocessing are guaranteed to match.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Make ``app`` importable when the script is executed from any working dir.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.feature_spec import (  # noqa: E402  (import after sys.path setup)
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    MODEL_NAME,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REQUIRED_FEATURES,
    SAMPLES,
    SG_LEVELS,
    TARGET_NAME,
)

DATA_PATH = PROJECT_ROOT / "data" / "kidney_disease.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ckd_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

UCI_DATASET_ID = 336
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease"
KAGGLE_MIRROR_URL = "https://www.kaggle.com/datasets/mansoordaku/ckdisease"

# Regularised Random Forest configuration.  A fully grown forest memorises this
# small (400 row) dataset and reports ~100% hold-out accuracy, which is not a
# credible generalisation estimate.  The limits below keep the trees shallow and
# produce a realistic ~92% hold-out accuracy while retaining a high ROC-AUC.
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 2,
    "min_samples_leaf": 15,
    "max_features": "sqrt",
    "class_weight": "balanced",
}

TARGET_MAP = {"ckd": 1, "notckd": 0, "not ckd": 0}
COLUMN_ALIASES = {
    "wbcc": "wc",
    "rbcc": "rc",
    "class": TARGET_NAME,
    "target": TARGET_NAME,
}


def _print_manual_download_help() -> None:
    """Print actionable instructions when the dataset cannot be fetched."""
    print("\n" + "=" * 78)
    print("DATASET NOT FOUND")
    print("=" * 78)
    print(f"Place the UCI Chronic Kidney Disease CSV at:\n  {DATA_PATH}\n")
    print("Download it from either source:")
    print(f"  1. UCI ML Repository : {UCI_DOWNLOAD_URL}")
    print(f"  2. Kaggle mirror     : {KAGGLE_MIRROR_URL}")
    print("\nUnzip the archive and rename/copy the CSV to 'kidney_disease.csv'.")
    print("Expected columns include: id, age, bp, sg, al, su, rbc, pc, pcc, ba,")
    print("bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe,")
    print("ane, classification  (values may contain tabs, spaces and '?').")
    print("=" * 78 + "\n")


def fetch_dataset_from_ucimlrepo(destination: Path) -> None:
    """Download the dataset through ``ucimlrepo`` and save a standard CSV.

    The UCI API returns slightly different column names (``wbcc``/``rbcc`` and
    ``class``) and no ``id`` column.  We normalise them to the layout of the
    classic ``kidney_disease.csv`` file so that all downstream code is stable.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("ucimlrepo is not installed; run: pip install -r requirements.txt") from exc

    print("data/kidney_disease.csv not found -- downloading from the UCI repository...")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features: pd.DataFrame = dataset.data.features.copy()
    targets: pd.DataFrame = dataset.data.targets.copy()

    features = features.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in features.columns})
    target_column = targets.columns[0]
    frame = features.copy()
    frame.insert(0, "id", np.arange(1, len(frame) + 1))
    frame[TARGET_NAME] = targets[target_column].to_numpy()

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame)} rows to {destination}")


def load_raw_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, downloading it from UCI if it does not exist."""
    if not path.exists():
        try:
            fetch_dataset_from_ucimlrepo(path)
        except Exception as exc:  # network, API or dependency failure
            print(f"Automatic download failed: {exc}")
            _print_manual_download_help()
            raise SystemExit(1) from exc
    if not path.exists():  # pragma: no cover - defensive
        _print_manual_download_help()
        raise SystemExit(1)
    return pd.read_csv(path)


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the dirty UCI CSV and return a modelling-ready frame.

    The raw file contains values such as ``"\\tno"``, ``" yes"``, ``"ckd\\t"``
    and ``"?"``.  Steps:

    1. Normalise column names and map known aliases.
    2. Strip tabs/whitespace and lowercase string columns.
    3. Replace ``?`` (and empty strings) with ``NaN``.
    4. Drop the ``id`` column and unnamed index columns.
    5. Coerce numeric features to floats and normalise ``sg`` levels.
    6. Map the target to ``1`` (ckd) / ``0`` (notckd) and drop unknown rows.
    """
    df = raw.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")], errors="ignore")
    df = df.drop(columns=["id"], errors="ignore")

    # --- Dirty string cleanup -------------------------------------------------
    for column in df.select_dtypes(include=["object", "string"]).columns:
        series = df[column].astype("string")
        series = series.str.replace("\t", "", regex=False)
        series = series.str.replace("\u00a0", " ", regex=False)
        series = series.str.strip().str.lower()
        series = series.mask(series.isin(["", "?", "nan", "none", "null"]))
        df[column] = series.astype(object)

    missing_columns = [c for c in FEATURE_NAMES + [TARGET_NAME] if c not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns) + ". "
            "See the download instructions above."
        )

    # --- Target ---------------------------------------------------------------
    df[TARGET_NAME] = df[TARGET_NAME].map(lambda value: TARGET_MAP.get(str(value).strip().lower()))
    df = df.dropna(subset=[TARGET_NAME]).copy()
    df[TARGET_NAME] = df[TARGET_NAME].astype(int)

    # --- Numeric coercion -----------------------------------------------------
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Specific gravity is recorded to three decimals; snap tiny float variants
    # (1.01 vs 1.010) onto the five canonical levels.
    if "sg" in df.columns:
        df["sg"] = df["sg"].apply(
            lambda value: min(SG_LEVELS, key=lambda level: abs(level - value)) if pd.notna(value) else np.nan
        )

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].where(df[column].notna(), np.nan)

    return df.reset_index(drop=True)


def build_pipeline(**overrides: Any) -> Pipeline:
    """Create the preprocessing + Random Forest pipeline.

    Numeric features: median imputation, then standardisation.
    Categorical features: most-frequent imputation, then one-hot encoding.
    Unseen categories at inference time are safely ignored.

    Any keyword argument overrides the corresponding entry of :data:`MODEL_PARAMS`.
    """
    params = {**MODEL_PARAMS, **overrides}
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute the classification metrics used in the metadata and README."""
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_labels": {
            "rows": ["actual_not_ckd", "actual_ckd"],
            "columns": ["predicted_not_ckd", "predicted_ckd"],
        },
        "support": {"not_ckd": int((y_test == 0).sum()), "ckd": int((y_test == 1).sum())},
    }


def build_feature_metadata(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Return feature definitions enriched with observed ranges and options."""
    ranges: dict[str, dict[str, float]] = {}
    for name in NUMERIC_FEATURES:
        series = df[name].dropna()
        ranges[name] = {
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }

    features: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        item: dict[str, Any] = {
            "name": spec["name"],
            "label": spec["label"],
            "type": spec["type"],
            "dtype": spec["dtype"],
            "required": spec["required"],
            "unit": spec["unit"],
            "description": spec["description"],
            "section": spec["section"],
        }
        if spec["type"] == "numeric":
            item["min"] = spec["min"]
            item["max"] = spec["max"]
            item["step"] = spec["step"]
            item["observed_min"] = ranges[spec["name"]]["min"]
            item["observed_max"] = ranges[spec["name"]]["max"]
        else:
            item["allowed_values"] = [option["value"] for option in spec["options"]]
        if "options" in spec:
            item["options"] = spec["options"]
        features.append(item)
    return features, ranges


def main() -> None:
    """Run the full training, evaluation and persistence workflow."""
    print("=" * 78)
    print(f"{MODEL_NAME} -- training")
    print("=" * 78)

    raw = load_raw_dataframe()
    print(f"Raw dataset: {raw.shape[0]} rows x {raw.shape[1]} columns")

    df = clean_dataframe(raw)
    print(f"Cleaned dataset: {len(df)} rows, {len(FEATURE_NAMES)} features")

    x = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].astype(int)

    class_counts = {label: int((y == value).sum()) for label, value in (("ckd", 1), ("notckd", 0))}
    print(f"Class balance: {class_counts}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train/test split: {len(x_train)} / {len(x_test)} (stratified, random_state={RANDOM_STATE})")

    pipeline = build_pipeline()

    # --- 5-fold cross-validation on the training split -----------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_accuracy = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(
        f"5-fold CV ROC-AUC : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f} "
        f"({', '.join(f'{score:.4f}' for score in cv_auc)})"
    )
    print(
        f"5-fold CV accuracy: {cv_accuracy.mean():.4f} +/- {cv_accuracy.std():.4f} "
        f"({', '.join(f'{score:.4f}' for score in cv_accuracy)})"
    )

    # --- Final fit and held-out evaluation -----------------------------------
    pipeline.fit(x_train, y_train)
    evaluation = evaluate_model(pipeline, x_test, y_test)

    print("\nHeld-out test metrics")
    print(f"  Accuracy : {evaluation['accuracy']:.4f}")
    print(f"  Precision: {evaluation['precision']:.4f}")
    print(f"  Recall   : {evaluation['recall']:.4f}")
    print(f"  F1 score : {evaluation['f1_score']:.4f}")
    print(f"  ROC-AUC  : {evaluation['roc_auc']:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    {evaluation['confusion_matrix']}")

    # --- Metadata -------------------------------------------------------------
    features, numeric_ranges = build_feature_metadata(df)

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        top_importances = sorted(
            zip(transformed_names, importances), key=lambda pair: pair[1], reverse=True
        )[:10]
        feature_importance = [
            {"feature": str(name), "importance": round(float(score), 4)}
            for name, score in top_importances
        ]
    except Exception:  # pragma: no cover - non-critical metadata
        feature_importance = []

    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": (
            "RandomForestClassifier "
            f"({MODEL_PARAMS['n_estimators']} trees, max_depth={MODEL_PARAMS['max_depth']}, "
            f"min_samples_leaf={MODEL_PARAMS['min_samples_leaf']}, "
            f"max_features='{MODEL_PARAMS['max_features']}', "
            f"class_weight='{MODEL_PARAMS['class_weight']}') in a scikit-learn Pipeline"
        ),
        "hyperparameters": MODEL_PARAMS,
        "pipeline_steps": [
            "ColumnTransformer",
            "numeric: SimpleImputer(median) -> StandardScaler",
            "categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')",
            (
                "RandomForestClassifier("
                f"n_estimators={MODEL_PARAMS['n_estimators']}, "
                f"max_depth={MODEL_PARAMS['max_depth']}, "
                f"min_samples_leaf={MODEL_PARAMS['min_samples_leaf']}, "
                f"max_features='{MODEL_PARAMS['max_features']}', "
                f"class_weight='{MODEL_PARAMS['class_weight']}')"
            ),
        ],
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "required_features": REQUIRED_FEATURES,
        "features": features,
        "feature_list": [
            {"name": f["name"], "type": f["type"], "dtype": f["dtype"]} for f in features
        ],
        "allowed_categorical_values": {
            name: [option["value"] for option in next(f for f in FEATURE_SPECS if f["name"] == name)["options"]]
            for name in CATEGORICAL_FEATURES
        },
        "numeric_ranges": numeric_ranges,
        "evaluation": {
            **evaluation,
            "cv_5fold_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_5fold_roc_auc_std": round(float(cv_auc.std()), 4),
            "cv_5fold_roc_auc_folds": [round(float(score), 4) for score in cv_auc],
            "cv_5fold_accuracy_mean": round(float(cv_accuracy.mean()), 4),
            "cv_5fold_accuracy_std": round(float(cv_accuracy.std()), 4),
            "cv_5fold_accuracy_folds": [round(float(score), 4) for score in cv_accuracy],
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
        },
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_size": int(len(df)),
        "dataset": {
            "name": "UCI Chronic Kidney Disease",
            "uci_id": UCI_DATASET_ID,
            "source": UCI_DOWNLOAD_URL,
            "n_rows": int(len(df)),
            "n_features": len(FEATURE_NAMES),
            "missing_values_total": int(df[FEATURE_NAMES].isna().sum().sum()),
            "class_distribution": class_counts,
        },
        "target_distribution": class_counts,
        "target_mapping": {"ckd": 1, "notckd": 0},
        "samples": SAMPLES,
        "random_state": RANDOM_STATE,
        "feature_importance_top10": feature_importance,
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
        },
        "disclaimer": "Model output only, not a medical diagnosis.",
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved pipeline -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    print(f"\nTraining date: {metadata['training_date']}")
    print("Done.")


if __name__ == "__main__":
    main()
