"""Shared paths and scientific constants; all artifacts stay in this workspace."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "oncosentry_data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"
SEED = 42
CLASS_NAMES = {
    "BRCA": "Breast invasive carcinoma",
    "COAD": "Colon adenocarcinoma",
    "KIRC": "Kidney renal clear cell carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "PRAD": "Prostate adenocarcinoma",
}
CLASSES = sorted(CLASS_NAMES)
EXPECTED_COUNTS = {"BRCA": 300, "COAD": 78, "KIRC": 146, "LUAD": 141, "PRAD": 136}
FEATURE_COUNTS = [10, 25, 50, 100, 500]
DISCLAIMER = "This project is a prototype and is not medical advice."

