"""Reproducible synthetic measurement experiments on the locked test split.

Noise is additive Gaussian noise in the supplied expression scale. Its standard
deviation is ``level * training_gene_standard_deviation``. Missing entries are
sampled independently and left as NaN for each saved model's training-fitted
imputer. Neither simulation establishes clinical laboratory robustness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, RESULT_DIR, SEED

NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
MISSING_LEVELS = (0.0, 0.05, 0.1, 0.2)
EXPERIMENT_SEEDS = (42, 137, 2026)


def _numeric_values(values: np.ndarray) -> np.ndarray:
    """Copy one vector or a sample-by-feature matrix, preserving existing NaNs."""
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or array.shape[-1] == 0:
        raise ValueError("Expected a nonempty vector or sample-by-feature matrix.")
    if np.isinf(array).any():
        raise ValueError("Infinite expression values are not permitted.")
    return array.copy()


def add_noise(values: np.ndarray, training_std: np.ndarray, level: float,
              seed: int = SEED) -> np.ndarray:
    """Return an unchanged-size copy with training-scaled Gaussian noise.

    Zero training standard deviation gives zero perturbation for that gene.
    Values are deliberately not clipped: clipping would change the experiment.
    Negative simulated measurements are possible and may be unrealistic.
    """
    array = _numeric_values(values)
    scale = np.asarray(training_std, dtype=float)
    if scale.ndim != 1 or len(scale) != array.shape[-1]:
        raise ValueError("Training standard deviations must match the feature order.")
    if not np.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("Training standard deviations must be finite and nonnegative.")
    if not np.isfinite(level) or level < 0:
        raise ValueError("Noise level must be finite and nonnegative.")
    rng = np.random.default_rng(seed)
    return array + rng.normal(size=array.shape) * scale * level


def mask_missing(values: np.ndarray, fraction: float, seed: int = SEED) -> np.ndarray:
    """Independently mask each measurement with the requested probability.

    The fraction is an expected rate, so finite samples need not contain exactly
    that percentage of NaNs. A shared seed gives nested masks across fractions.
    """
    array = _numeric_values(values)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Missingness fraction must be between zero and one.")
    rng = np.random.default_rng(seed)
    array[rng.random(array.shape) < fraction] = np.nan
    return array


def _require_locked_protocol() -> dict:
    """Prevent exploratory use of the final test split before protocol lock."""
    path = RESULT_DIR / "protocol_lock.json"
    if not path.exists():
        raise RuntimeError("Run training and lock the evaluation protocol first.")
    protocol = json.loads(path.read_text())
    if not protocol.get("locked", False) or not protocol.get("test_evaluation_authorized", False):
        raise RuntimeError("The evaluation protocol must be locked and final test evaluation authorized.")
    return protocol


def run_experiments(model_ids: list[str] | None = None) -> pd.DataFrame:
    """Evaluate the preregistered perturbations without changing any model.

    Perturbations are generated in original feature order before selecting each
    model's columns. All models therefore see identical errors for a shared gene.
    The saved training split alone supplies the noise scaling statistics.
    """
    from src.evaluate_models import evaluate_predictions
    from src.validate_data import load_dataset

    _require_locked_protocol()
    registry = json.loads((MODEL_DIR / "registry.json").read_text())
    manifests = {m["model_id"]: m for m in registry["models"]}
    if model_ids is None:
        full_id = next(m["model_id"] for m in registry["models"]
                       if m["n_features"] == 20531 and m["model_id"] != "majority")
        model_ids = list(dict.fromkeys([full_id, "compact_10", registry["selected_model"]]))
    if not model_ids or any(model_id not in manifests for model_id in model_ids):
        raise ValueError("Every requested model must occur in the saved model registry.")
    split = json.loads((RESULT_DIR / "split_manifest.json").read_text())["splits"]
    X, y, _ = load_dataset()
    training_std = X.loc[split["train"]].std(axis=0, ddof=0).to_numpy(dtype=float)
    X_test = X.loc[split["test"]]
    values = X_test.to_numpy(dtype=float)
    labels = y.loc[X_test.index].to_numpy()
    columns = {name: i for i, name in enumerate(X.columns)}
    models = {}
    for model_id in model_ids:
        manifest = manifests[model_id]
        models[model_id] = (
            joblib.load(MODEL_DIR / manifest["artifact"]),
            [columns[name] for name in manifest["input_features"]],
            float(manifest.get("review_threshold", 0.8)),
        )
    records, errors = [], []
    for experiment, levels in (("noise", NOISE_LEVELS), ("missingness", MISSING_LEVELS)):
        for level in levels:
            for seed in EXPERIMENT_SEEDS:
                perturbed = (add_noise(values, training_std, level, seed)
                             if experiment == "noise" else mask_missing(values, level, seed))
                for model_id, (model, indices, threshold) in models.items():
                    inputs = perturbed[:, indices]
                    predicted = model.predict(inputs)
                    probabilities = model.predict_proba(inputs)
                    measured = evaluate_predictions(labels, predicted, probabilities)
                    records.append({"model_id": model_id, "experiment": experiment,
                                    "level": level, "seed": seed, **measured["summary"]})
                    for row_index in np.flatnonzero(predicted != labels):
                        scores = probabilities[row_index]
                        error = {
                            "model_id": model_id, "experiment": experiment,
                            "level": level, "seed": seed,
                            "sample_id": X_test.index[row_index],
                            "true_label": labels[row_index],
                            "predicted_label": predicted[row_index],
                            "max_score": float(scores.max()),
                            "uncertain": bool(scores.max() < threshold),
                            "explanation": "Observed error under a synthetic perturbation; biological cause unknown.",
                        }
                        error.update({f"score_{name}": float(score)
                                      for name, score in zip(model.classes_, scores)})
                        errors.append(error)
    output = RESULT_DIR / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "robustness.csv", index=False)
    error_columns = ["model_id", "experiment", "level", "seed", "sample_id",
                     "true_label", "predicted_label", "max_score", "uncertain", "explanation"]
    pd.DataFrame(errors, columns=error_columns + [f"score_{name}" for name in models[model_ids[0]][0].classes_]).to_csv(
        output / "robustness_errors.csv", index=False)
    methodology = {
        "split": "locked final test", "sample_count": len(labels),
        "models": model_ids, "seeds": list(EXPERIMENT_SEEDS),
        "noise_levels": list(NOISE_LEVELS), "missingness_levels": list(MISSING_LEVELS),
        "noise": "Independent additive N(0, (level * training population std)^2) per measurement; no clipping.",
        "missingness": "Independent Bernoulli masking; NaNs passed to saved training-median imputers.",
        "comparison": "Shared full-dimensional perturbations before model-specific feature slicing.",
        "scope": "Synthetic sensitivity analysis; not a simulation validated against clinical laboratory errors.",
        "replicates": "Seeds repeat perturbations on the same samples, not independent patient cohorts.",
        "selection": "Results do not change features, model choice, or review thresholds.",
    }
    (output / "robustness_methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Saved model IDs; default full, 10-gene, selected model.")
    args = parser.parse_args()
    frame = run_experiments(args.models)
    print(frame.groupby(["model_id", "experiment", "level"])["macro_f1"].agg(["mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
