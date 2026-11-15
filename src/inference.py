
"""Load trusted local artifacts and run the fitted model on its exact input schema."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR
from src.manual_inputs import validate_manual_values
from src.uncertainty import outside_training_ranges, score_uncertainty


def load_registry(model_dir: Path = MODEL_DIR) -> dict:
    """Read the model catalogue; missing artifacts yield a useful actionable error."""
    path = Path(model_dir) / "registry.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Trained artifacts are not available. Run the training command in README.md first."
        )
    registry = json.loads(path.read_text())
    if not isinstance(registry.get("models"), list):
        raise ValueError("The model registry must contain a models list.")
    return registry


def load_bundle(model_id: str, model_dir: Path = MODEL_DIR) -> dict:
    """Load a locally trained pipeline, its manifest, and ordered feature metadata.

    Joblib files are executable Python serialization. Only use artifacts produced
    by this project, never a model uploaded by an untrusted visitor.
    """
    model_dir = Path(model_dir)
    registry = load_registry(model_dir)
    matches = [m for m in registry["models"] if m["model_id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous model: {model_id}")

    manifest = matches[0]
    model = joblib.load(model_dir / manifest["artifact"])
    features = pd.read_csv(
        model_dir / manifest["selected_features_file"],
        keep_default_na=False
    )

    names = manifest["input_features"]

    if len(names) != len(set(names)) or list(features["uci_feature"]) != names:
        raise ValueError(
            "Saved feature metadata does not match the model input schema."
        )

    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(names):
        raise ValueError(
            "The pipeline expects a different number of measurements."
        )

    if list(model.classes_) != list(manifest["classes"]):
        raise ValueError(
            "Class order differs between the model and manifest."
        )

    return {
        "model": model,
        "manifest": manifest,
        "features": features
    }


def predict_values(
    bundle: dict,
    values,
    *,
    allow_missing: bool = False
) -> dict:
    """Run one sample through its saved preprocessing and classifier.

    Manual inputs require complete values. The separate robustness experiment
    deliberately permits NaN and uses only the saved training-fitted imputer.
    """

    manifest = bundle["manifest"]
    model = bundle["model"]
    features = bundle["features"]

    names = manifest["input_features"]

    if isinstance(values, Mapping):
        array = validate_manual_values(values, names)
    else:
        array = np.asarray(values, dtype=float)

        if array.ndim == 1:
            array = array.reshape(1, -1)

    if array.shape != (1, len(names)):
        raise ValueError(
            f"Expected one sample with exactly {len(names)} ordered gene measurements."
        )

    if np.isinf(array).any() or (
        not allow_missing and not np.isfinite(array).all()
    ):
        raise ValueError(
            "Measurements must be finite; missing values are only allowed "
            "in the robustness experiment."
        )

    # Generate genuine class probabilities using the saved trained model.
    scores = np.asarray(
        model.predict_proba(array)[0],
        dtype=float
    )

    # Read the validation-derived review threshold.
    threshold = float(manifest.get("review_threshold", 1.0))

    # The majority-class baseline may have no qualifying threshold.
    # Preserve that fact without changing the saved model registry.
    if not 0.0 <= threshold <= 1.0:
        description = score_uncertainty(scores, 1.0)
        description["uncertain"] = True
        description["review_rule_unavailable"] = True
    else:
        description = score_uncertainty(scores, threshold)
        description["review_rule_unavailable"] = False

    # Check whether input measurements fall outside training ranges.
    out_of_range = outside_training_ranges(
        array,
        features["train_min"],
        features["train_max"],
        names
    )

    # Identify the highest-scoring cancer category.
    predicted = str(
        model.classes_[int(np.argmax(scores))]
    )

    # Return prediction information to the Streamlit website.
    return {
        **description,
        "model_id": manifest["model_id"],
        "predicted_label": predicted,
        "probabilities": {
            str(label): float(score)
            for label, score in zip(model.classes_, scores)
        },
        "out_of_range": out_of_range,
        "missing_count": int(np.isnan(array).sum()),
        "review_recommended": bool(
            description["uncertain"]
            or out_of_range
            or np.isnan(array).any()
        ),
    }
