"""Fit on training data, select with validation, lock, then evaluate test once."""
from __future__ import annotations
import argparse
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from config import CLASSES, FEATURE_COUNTS, MODEL_DIR, RESULT_DIR, SEED
from src.evaluate_models import evaluate_predictions
from src.preprocess import make_pipeline, compact_pipeline
from src.split_data import create_split, load_split
from src.utilities import ensure_directories, write_json, utc_now, environment_versions, sha256
from src.validate_data import load_dataset


def validation_review_threshold(y, probabilities):
    """Exploratory validation rule, not a clinically calibrated decision threshold."""
    best = probabilities.max(axis=1)
    correct = np.asarray(CLASSES)[probabilities.argmax(axis=1)] == np.asarray(y)
    for threshold in np.round(np.arange(0.50, 1.0, 0.01), 2):
        retained = best >= threshold
        if retained.mean() >= 0.5 and correct[retained].mean() >= 0.95:
            return float(threshold), {"retained": int(retained.sum()), "coverage": float(retained.mean()),
                                      "retained_accuracy": float(correct[retained].mean())}
    return 1.01, {"retained": 0, "coverage": 0.0, "retained_accuracy": None}


def calibration_bins(y, probabilities, model_id):
    """Top-label reliability bins on validation data, without fitting calibration."""
    best = probabilities.max(axis=1)
    correct = np.asarray(CLASSES)[probabilities.argmax(axis=1)] == np.asarray(y)
    rows = []
    for i in range(10):
        mask = (best >= i / 10) & ((best < (i + 1) / 10) if i < 9 else (best <= 1))
        if mask.any():
            rows.append({"model_id": model_id, "split": "validation", "bin_lower": i / 10,
                         "bin_upper": (i + 1) / 10, "n_samples": int(mask.sum()),
                         "mean_score": float(best[mask].mean()), "observed_accuracy": float(correct[mask].mean())})
    return rows


def prediction_rows(model_id, ids, y, probabilities):
    """Persist each actual prediction so errors and aggregate metrics are auditable."""
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    rows = []
    for i, sid in enumerate(ids):
        row = {"model_id": model_id, "sample_id": sid, "true_label": y.loc[sid],
               "predicted_label": predicted[i], "correct": bool(predicted[i] == y.loc[sid]),
               "max_score": float(probabilities[i].max())}
        row.update({f"score_{label}": float(probabilities[i, j]) for j, label in enumerate(CLASSES)})
        rows.append(row)
    return rows


def append_evaluation(summary, per_class, matrices, predictions, manifest, split, ids, y, probabilities):
    model_id = manifest["model_id"]
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    result = evaluate_predictions(y.loc[ids], predicted, probabilities)
    summary.append({"model_id": model_id, "split": split, "n_features": manifest["n_features"],
                    "model_bytes": manifest["model_bytes"], **result["summary"]})
    per_class.extend({"model_id": model_id, "split": split, **row} for row in result["per_class"])
    matrices.setdefault(model_id, {})[split] = result["confusion_matrix"]
    predictions.extend(prediction_rows(model_id, ids, y, probabilities))
    return result["summary"]


def save_evaluations(summary, per_class, matrices, predictions, split):
    pd.DataFrame(summary).to_csv(RESULT_DIR / "metrics/summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(RESULT_DIR / "metrics/per_class.csv", index=False)
    write_json(RESULT_DIR / "metrics/confusion_matrices.json", matrices)
    pd.DataFrame(predictions).to_csv(RESULT_DIR / f"predictions/{split}_predictions.csv", index=False)


def feature_table(cross, X_train, selected, scores):
    table = cross.iloc[selected].copy()
    values = X_train[:, selected]
    table["selection_score"] = scores
    for name, function in [("min", np.min), ("max", np.max), ("mean", np.mean),
                           ("std", np.std), ("median", np.median)]:
        table[f"train_{name}"] = function(values, axis=0)
    return table


def train():
    """Train every predeclared model without accessing test measurements for fitting."""
    ensure_directories()
    if (RESULT_DIR / "protocol_lock.json").exists():
        raise RuntimeError("This experiment is already locked. Preserve its results; do not tune on its test set.")
    X, y, cross = load_dataset()
    splits = create_split(X, y)
    train_ids, val_ids = splits["train"], splits["validation"]
    X_train = X.loc[train_ids].to_numpy()
    X_val = X.loc[val_ids].to_numpy()
    y_train = y.loc[train_ids].to_numpy()
    summary, per_class, matrices, predictions, manifests, calibration = [], [], {}, [], [], []
    definitions = [("majority", None), ("full", "all")] + [(f"compact_{k}", k) for k in FEATURE_COUNTS]
    for model_id, k in definitions:
        started = time.perf_counter()
        print(f"Training {model_id}: train={len(train_ids)}, validation={len(val_ids)}", flush=True)
        if model_id == "majority":
            selected = np.array([0])
            model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("classifier", DummyClassifier(strategy="most_frequent", random_state=SEED))])
            model.fit(X_train[:, selected], y_train)
            scores = np.array([0.0])
            research_file = None
        else:
            research = make_pipeline(k)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                research.fit(X_train, y_train)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise RuntimeError(f"{model_id} did not converge: {convergence}")
            model, selected = compact_pipeline(research, X_train)
            research_file = f"{model_id}_research.joblib"
            joblib.dump(research, MODEL_DIR / research_file, compress=3)
            scores = research.named_steps["selector"].scores_[selected]
            # Check exact export on validation as well as training.
            if not np.allclose(research.predict_proba(X_val), model.predict_proba(X_val[:, selected]),
                               atol=1e-12, rtol=1e-12):
                raise RuntimeError("Validation prediction parity failed.")
        artifact = f"{model_id}.joblib"
        joblib.dump(model, MODEL_DIR / artifact, compress=3)
        features_file = f"{model_id}_features.csv"
        table = feature_table(cross, X_train, selected, scores)
        table.to_csv(MODEL_DIR / features_file, index=False)
        probabilities = model.predict_proba(X_val[:, selected])
        reloaded = joblib.load(MODEL_DIR / artifact)
        if not np.array_equal(model.classes_, np.array(CLASSES)):
            raise RuntimeError("Model class order differs from manifest order.")
        if not np.allclose(probabilities, reloaded.predict_proba(X_val[:, selected]), atol=1e-12, rtol=1e-12):
            raise RuntimeError("Saved-model reload changed probabilities.")
        threshold, threshold_stats = validation_review_threshold(y.loc[val_ids], probabilities)
        manifest = {"model_id": model_id,
                    "display_name": "Majority-class baseline" if model_id == "majority" else
                                    ("Full-feature logistic regression" if model_id == "full" else f"{k}-gene logistic regression"),
                    "n_features": 0 if model_id == "majority" else len(selected),
                    "input_features": X.columns[selected].tolist(), "classes": CLASSES,
                    "artifact": artifact, "artifact_sha256": sha256(MODEL_DIR / artifact),
                    "research_artifact": research_file, "selected_features_file": features_file,
                    "model_bytes": (MODEL_DIR / artifact).stat().st_size,
                    "training_seconds": time.perf_counter() - started, "n_training_samples": len(train_ids),
                    "review_threshold": threshold, "threshold_validation": threshold_stats,
                    "threshold_note": "Validation-only exploratory threshold: lowest grid value retaining at least 50% of samples with at least 95% empirical accuracy. This small sample does not establish calibrated or clinical reliability; 1.01 means review all.",
                    "probability_status": "Uncalibrated model-estimated class probabilities, not cancer risk.",
                    "measurement_scale": "Original UCI supplied expression scale; raw counts/other assays are not interchangeable.",
                    "mapping_status": "PROVISIONAL third-party annotations; original TCGA index not independently verified.",
                    "fit_sample_ids": train_ids,
                    "parameters": {"classifier": "most_frequent"} if model_id == "majority" else
                                  {"C": 1.0, "solver": "lbfgs", "max_iter": 3000, "tol": 1e-4, "random_state": SEED}}
        manifest["validation"] = append_evaluation(summary, per_class, matrices, predictions, manifest,
                                                    "validation", val_ids, y, probabilities)
        calibration.extend(calibration_bins(y.loc[val_ids], probabilities, model_id))
        manifests.append(manifest)
        write_json(MODEL_DIR / f"{model_id}.json", manifest)
        save_evaluations(summary, per_class, matrices, predictions, "validation")
        print(f"{model_id}: accuracy={manifest['validation']['accuracy']:.4f}, macro-F1={manifest['validation']['macro_f1']:.4f}; {manifest['training_seconds']:.1f}s", flush=True)
    compact = [m for m in manifests if m["model_id"].startswith("compact_")]
    selected_model = sorted(compact, key=lambda m: (-m["validation"]["macro_f1"], m["n_features"]))[0]["model_id"]
    registry = {"schema_version": 1, "created_at_utc": utc_now(), "models": manifests,
                "selected_model": selected_model, "manual_model": "compact_10",
                "selection_rule": "Highest validation macro-F1 among compact models; ties use fewer features. Full model is a separate comparator.",
                "environment": environment_versions()}
    write_json(MODEL_DIR / "registry.json", registry)
    pd.DataFrame(calibration).to_csv(RESULT_DIR / "metrics/calibration.csv", index=False)
    write_json(RESULT_DIR / "environment.json", environment_versions())
    # Lock all choices before any held-out predictions or robustness experiments.
    write_json(RESULT_DIR / "protocol_lock.json", {
        "locked": True, "test_evaluation_authorized": True, "locked_at_utc": utc_now(),
        "selected_model": selected_model, "model_ids": [m["model_id"] for m in manifests],
        "selection_rule": registry["selection_rule"], "seed": SEED,
        "split_manifest_sha256": sha256(RESULT_DIR / "split_manifest.json"),
        "registry_sha256": sha256(MODEL_DIR / "registry.json"),
        "noise_levels": [0, .1, .25, .5, 1, 2], "missingness_levels": [0, .05, .1, .2],
        "perturbation_seeds": [42, 137, 2026],
        "test_policy": "Single final evaluation of predeclared models; robustness is descriptive, never used for retuning.",
        "no_train_validation_refit": True})
    # Gene comparisons use training data only, avoiding exploratory test leakage.
    top25 = next(m for m in manifests if m["model_id"] == "compact_25")["input_features"]
    expression = X.loc[train_ids, top25].copy()
    expression.insert(0, "cancer_type", y.loc[train_ids])
    expression.insert(1, "split", "train")
    expression.reset_index().melt(id_vars=["sample_id", "cancer_type", "split"],
                                 var_name="uci_feature", value_name="value").to_csv(
        RESULT_DIR / "experiments/gene_expression.csv", index=False)
    print(f"Protocol locked. Selected compact model: {selected_model}. Test measurements have not been used for fitting or selection.", flush=True)


def evaluate_test():
    """Evaluate the locked model collection once, with no model changes."""
    completion = RESULT_DIR / "test_evaluation.json"
    if completion.exists():
        raise RuntimeError("Final test evaluation already completed. Read saved results rather than repeatedly tuning against test data.")
    lock = json.loads((RESULT_DIR / "protocol_lock.json").read_text())
    if not lock.get("locked") or not lock.get("test_evaluation_authorized"):
        raise RuntimeError("A locked evaluation protocol is required.")
    if sha256(MODEL_DIR / "registry.json") != lock["registry_sha256"] or sha256(RESULT_DIR / "split_manifest.json") != lock["split_manifest_sha256"]:
        raise RuntimeError("Registry or split changed after protocol lock.")
    registry = json.loads((MODEL_DIR / "registry.json").read_text())
    X, y, _ = load_dataset()
    splits = load_split()
    ids = splits["test"]
    summary = pd.read_csv(RESULT_DIR / "metrics/summary.csv").to_dict("records")
    per_class = pd.read_csv(RESULT_DIR / "metrics/per_class.csv").to_dict("records")
    matrices = json.loads((RESULT_DIR / "metrics/confusion_matrices.json").read_text())
    predictions = []
    for manifest in registry["models"]:
        path = MODEL_DIR / manifest["artifact"]
        if sha256(path) != manifest["artifact_sha256"]:
            raise RuntimeError(f"Model artifact changed: {path.name}")
        model = joblib.load(path)
        probabilities = model.predict_proba(X.loc[ids, manifest["input_features"]].to_numpy())
        result = append_evaluation(summary, per_class, matrices, predictions, manifest, "test", ids, y, probabilities)
        print(f"TEST {manifest['model_id']}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
    save_evaluations(summary, per_class, matrices, predictions, "test")
    # Deterministic first three test identifiers per class, not chosen for prediction success.
    demo_ids = [sid for label in CLASSES for sid in y.loc[ids][y.loc[ids].eq(label)].index[:3]]
    demo = X.loc[demo_ids].copy()
    demo.insert(0, "cancer_type", y.loc[demo_ids])
    demo.insert(1, "split", "test")
    demo.to_csv(RESULT_DIR / "demo_samples.csv.gz", compression="gzip")
    write_json(completion, {"completed_at_utc": utc_now(), "n_samples": len(ids),
                           "model_ids": lock["model_ids"], "no_retuning": True,
                           "protocol_lock_sha256": sha256(RESULT_DIR / "protocol_lock.json"),
                           "demo_selection": "First three test sample IDs per class in split-manifest order, independent of predictions."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate saved locked models once.")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        evaluate_test() if args.evaluate_test else train()

