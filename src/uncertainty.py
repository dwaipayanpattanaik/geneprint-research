"""Descriptive model-score uncertainty; these are not clinical risk estimates."""
from __future__ import annotations

import numpy as np


def score_uncertainty(probabilities, threshold: float) -> dict:
    """Summarize one probability vector with a validation-selected review rule."""
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Expected a finite vector of class probabilities.")
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(), 1.0, atol=1e-6):
        raise ValueError("Class probabilities must lie in [0, 1] and sum to one.")
    if not 0 <= threshold <= 1:
        raise ValueError("The review threshold must lie in [0, 1].")
    ranked = np.sort(scores)
    positive = scores[scores > 0]
    return {
        "max_score": float(ranked[-1]),
        "margin": float(ranked[-1] - ranked[-2]),
        "entropy": float(-np.sum(positive * np.log(positive)) / np.log(len(scores))),
        "uncertain": bool(ranked[-1] < threshold),
    }


def outside_training_ranges(values, minimums, maximums, feature_names) -> list[str]:
    """List measurements outside per-gene training minima/maxima; ignore missingness."""
    values = np.asarray(values, dtype=float).reshape(-1)
    minimums = np.asarray(minimums, dtype=float)
    maximums = np.asarray(maximums, dtype=float)
    if len(values) != len(minimums) or len(values) != len(maximums) or len(values) != len(feature_names):
        raise ValueError("Range statistics must match the input schema.")
    return [str(name) for name, value, low, high in zip(feature_names, values, minimums, maximums)
            if np.isfinite(value) and (value < low or value > high)]
