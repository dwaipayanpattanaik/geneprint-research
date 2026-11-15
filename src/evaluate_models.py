"""Transparent multiclass evaluation with one-versus-rest FP and FN counts."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, log_loss)
from config import CLASSES


def evaluate_predictions(y, pred, proba=None):
    """Calculate metrics; confusion rows are true classes and columns predicted."""
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected nonempty equally sized one-dimensional labels.")
    if not set(y).issubset(CLASSES) or not set(pred).issubset(CLASSES):
        raise ValueError("Unknown tumor class.")
    cm = confusion_matrix(y, pred, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    summary = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
               "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
               "weighted_f1": f1_score(y, pred, labels=CLASSES, average="weighted", zero_division=0),
               "n_samples": len(y)}
    if proba is not None:
        proba = np.asarray(proba)
        if proba.shape != (len(y), len(CLASSES)) or not np.isfinite(proba).all():
            raise ValueError("Invalid probability shape or nonfinite scores.")
        if (proba < 0).any() or (proba > 1).any() or not np.allclose(proba.sum(axis=1), 1):
            raise ValueError("Class probabilities must sum to one.")
        summary["log_loss"] = log_loss(y, proba, labels=CLASSES)
        one_hot = (y[:, None] == np.array(CLASSES)[None, :]).astype(float)
        summary["multiclass_brier"] = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    rows = [{"class": label, "precision": precision[i], "recall": recall[i], "f1": f1[i],
             "support": int(support[i]), "false_positives": int(cm[:, i].sum() - cm[i, i]),
             "false_negatives": int(cm[i, :].sum() - cm[i, i])} for i, label in enumerate(CLASSES)]
    return {"summary": summary, "per_class": rows,
            "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()}}

