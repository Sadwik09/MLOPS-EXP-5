"""Load the trained pipeline and its metadata once at application startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.feature_spec import FEATURE_NAMES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "ckd_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"


class ModelLoader:
    """Owns the fitted pipeline and metadata for the lifetime of the app.

    Loading is fail-soft: if artifacts are missing or invalid the service still
    starts, ``/ready`` reports ``503`` and ``/predict`` returns ``503`` instead
    of crashing the process.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        metadata_path: Path | str = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._pipeline: Any | None = None
        self._metadata: dict[str, Any] | None = None
        self._load_error: str | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """Load artifacts from disk.

        Returns:
            ``True`` when both the pipeline and metadata are ready.
        """
        self._pipeline = None
        self._metadata = None
        self._load_error = None

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

            pipeline = joblib.load(self.model_path)
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

            if "model_version" not in metadata or "feature_names" not in metadata:
                raise ValueError("Metadata is missing 'model_version' or 'feature_names'.")

            trained_features = list(metadata.get("feature_names", []))
            if trained_features != FEATURE_NAMES:
                raise ValueError(
                    "Artifact feature schema does not match app.feature_spec. "
                    f"Trained: {trained_features}; expected: {FEATURE_NAMES}. "
                    "Re-run 'python train.py'."
                )

            self._pipeline = pipeline
            self._metadata = metadata
            logger.info(
                "Loaded model %s v%s (%s features)",
                metadata.get("model_name"),
                metadata.get("model_version"),
                metadata.get("n_features"),
            )
            return True
        except Exception as exc:  # noqa: BLE001 - expose a friendly readiness error
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load model artifacts: %s", exc)
            return False

    # ------------------------------------------------------------- properties
    @property
    def ready(self) -> bool:
        """Whether both the pipeline and metadata are available."""
        return self._pipeline is not None and self._metadata is not None

    @property
    def pipeline(self) -> Any:
        """The fitted scikit-learn pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")
        return self._pipeline

    @property
    def metadata(self) -> dict[str, Any]:
        """The model metadata dictionary."""
        if self._metadata is None:
            raise RuntimeError("Model metadata is not loaded.")
        return self._metadata

    @property
    def load_error(self) -> str | None:
        """Human-readable reason the last load failed, if any."""
        return self._load_error

    # ---------------------------------------------------------------- predict
    def predict(self, patient: dict[str, Any]) -> tuple[str, float, float]:
        """Run one prediction using the exact training-time pipeline.

        Args:
            patient: Mapping of feature name to value; missing keys become NaN.

        Returns:
            ``(label, probability_ckd, probability_not_ckd)`` where ``label`` is
            either ``"CKD"`` or ``"Not CKD"``.
        """
        row = {name: patient.get(name) for name in FEATURE_NAMES}
        frame = pd.DataFrame([row], columns=FEATURE_NAMES)

        # ``None`` must become a real NaN, otherwise scikit-learn's imputers
        # treat it as a legitimate category.
        frame = frame.where(pd.notna(frame), np.nan)
        for column in NUMERIC_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        probabilities = self.pipeline.predict_proba(frame)[0]
        classes = list(self.pipeline.classes_)
        ckd_index = classes.index(1) if 1 in classes else 0

        probability_ckd = float(probabilities[ckd_index])
        probability_not_ckd = float(1.0 - probability_ckd)
        label = "CKD" if probability_ckd >= 0.5 else "Not CKD"
        return label, probability_ckd, probability_not_ckd
