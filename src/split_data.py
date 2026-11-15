"""Create and persist a stratified split before fitting any transformation."""
import json
from sklearn.model_selection import train_test_split
from config import RESULT_DIR, SEED
from src.utilities import write_json, utc_now


def create_split(X, y):
    """Return 560 training, 120 validation, 121 test identifiers reproducibly."""
    train_val, test = train_test_split(X.index.to_numpy(), test_size=121,
                                     stratify=y.to_numpy(), random_state=SEED)
    train, validation = train_test_split(train_val, test_size=120,
                                        stratify=y.loc[train_val], random_state=SEED)
    splits = {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}
    if len(set(train) | set(validation) | set(test)) != len(X):
        raise ValueError("Split overlap or lost samples.")
    write_json(RESULT_DIR / "split_manifest.json", {
        "random_state": SEED, "created_at_utc": utc_now(), "splits": splits,
        "class_counts": {name: y.loc[ids].value_counts().to_dict() for name, ids in splits.items()},
        "policy": "Fit on training only; choose with validation; lock before test evaluation."})
    return splits


def load_split():
    return json.loads((RESULT_DIR / "split_manifest.json").read_text())["splits"]

