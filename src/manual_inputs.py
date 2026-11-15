"""Validate the exact compact-model input schema without inventing measurements."""
from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence

import numpy as np


MAX_ABSOLUTE_INPUT = 1e12


def validate_manual_values(values: Mapping, expected_features: Sequence[str]) -> np.ndarray:
    """Return one ordered numeric row; reject incomplete or ambiguous inputs.

    The accepted scale is the supplied research dataset scale. This function
    performs no normalization, clipping, rounding, or missing-value substitution.
    """
    expected = list(expected_features)
    if len(expected) != len(set(expected)):
        raise ValueError("The saved model schema contains duplicate features.")
    missing = sorted(set(expected) - set(values))
    extra = sorted(set(values) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(map(str, extra)))
        raise ValueError("The columns must exactly match the selected genes (" + "; ".join(details) + ").")
    ordered = []
    for feature in expected:
        raw = values[feature]
        if raw is None or isinstance(raw, bool) or str(raw).strip() == "":
            raise ValueError(f"{feature}: enter a numeric measurement; the field is empty or invalid.")
        try:
            value = float(raw)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{feature}: enter a numeric measurement.") from error
        if not np.isfinite(value):
            raise ValueError(f"{feature}: the value must be finite (no NaN or infinity).")
        if abs(value) > MAX_ABSOLUTE_INPUT:
            raise ValueError(f"{feature}: magnitude exceeds the supported numerical limit of 1e12. Check the measurement scale.")
        ordered.append(value)
    return np.asarray(ordered, dtype=float).reshape(1, -1)


def parse_manual_csv(content: bytes | str, expected_features: Sequence[str]) -> tuple[np.ndarray, bool]:
    """Read exactly one research sample and return (ordered values, was_reordered).

    Duplicate headers are detected before pandas could rename them. Correct but
    reordered columns are explicitly aligned by their original gene identifiers.
    """
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("Upload a UTF-8 CSV file.") from error
    if len(content) > 1_000_000:
        raise ValueError("Upload a one-row CSV smaller than 1 MB.")
    try:
        rows = list(csv.reader(io.StringIO(content), strict=True))
    except csv.Error as error:
        raise ValueError("The file is not a valid comma-separated CSV.") from error
    if len(rows) != 2:
        raise ValueError("The CSV must have one header row and exactly one data row.")
    header, row = rows
    if len(header) != len(set(header)):
        raise ValueError("Duplicate gene columns are not allowed.")
    if len(header) != len(row):
        raise ValueError("The data row must have one value for every header column.")
    values = validate_manual_values(dict(zip(header, row)), expected_features)
    return values, header != list(expected_features)


def make_csv_template(expected_features: Sequence[str], values: Sequence[float] | None = None) -> str:
    """Create a single-row CSV using the exact required original feature IDs."""
    features = list(expected_features)
    row = list(values) if values is not None else [""] * len(features)
    if len(row) != len(features):
        raise ValueError("Template values must match the number of selected genes.")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(features)
    writer.writerow(row)
    return output.getvalue()
