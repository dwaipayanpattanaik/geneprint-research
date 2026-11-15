"""Small reusable helpers for auditable local research artifacts."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from config import MODEL_DIR, RESULT_DIR, REPORT_DIR


def ensure_directories():
    """Create only derived-output directories, never touch source data."""
    for path in [MODEL_DIR, REPORT_DIR, *(RESULT_DIR / n for n in
                 ["metrics", "predictions", "charts", "experiments", "logs"])]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value):
    """Write portable JSON, converting NumPy scalars without allowing NaNs."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    """Hash a file incrementally so the archive is not loaded into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_versions() -> dict:
    """Record the actual runtime and packages used for reproducibility."""
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                "seaborn", "plotly", "streamlit", "joblib", "pytest"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

