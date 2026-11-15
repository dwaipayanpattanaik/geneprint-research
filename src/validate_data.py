"""Read-only validation of the prepared data and existing archive; no downloads."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import tarfile
import zipfile
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
from config import DATA_DIR, EXPECTED_COUNTS, RESULT_DIR
from src.gene_mapping import load_crosswalk
from src.utilities import sha256, write_json, utc_now


def load_dataset():
    """Return X indexed by sample_id, aligned y, and auditable crosswalk metadata."""
    cross = load_crosswalk()
    frame = pd.read_csv(DATA_DIR / "ONCOSENTRY_gene_named_801x20531.csv.gz")
    if frame.columns.tolist() != ["sample_id", "cancer_type"] + cross.output_column.tolist():
        raise ValueError("Prepared expression columns do not match the crosswalk in order.")
    ids = frame.pop("sample_id")
    labels = frame.pop("cancer_type")
    if len(ids) != 801 or ids.isna().any() or ids.duplicated().any():
        raise ValueError("Expected 801 unique, nonmissing sample identifiers.")
    if set(ids) != {f"sample_{i}" for i in range(801)}:
        raise ValueError("Unexpected or missing original sample identifiers.")
    if labels.value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Class counts differ from the expected public data.")
    values = frame.to_numpy(dtype=np.float64)
    if values.shape != (801, 20531) or not np.isfinite(values).all():
        raise ValueError("Expression data must have 801 by 20,531 finite numeric values.")
    X = pd.DataFrame(values, index=pd.Index(ids, name="sample_id"), columns=cross.uci_feature)
    X.columns.name = None
    y = pd.Series(labels.to_numpy(), index=X.index, name="cancer_type")
    if X.duplicated().any():
        raise ValueError("Identical expression profiles require investigation before splitting.")
    return X, y, cross


def validate_original_archive(X, y):
    """Compare all values and labels with the existing nested archive, in memory."""
    original_ids, labels = [], {}
    with zipfile.ZipFile(DATA_DIR / "UCI_original_download.zip") as outer:
        name = next(n for n in outer.namelist() if n.endswith(".tar.gz"))
        with outer.open(name) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                base = PurePosixPath(member.name).name
                if not member.isfile() or base not in {"data.csv", "labels.csv"}:
                    continue
                with archive.extractfile(member) as raw:
                    reader = csv.reader(line.decode("utf-8-sig") for line in raw)
                    header = next(reader)
                    if base == "data.csv":
                        if header[1:] != X.columns.tolist():
                            raise ValueError("Original feature order differs.")
                        for row in reader:
                            original_ids.append(row[0])
                            values = np.asarray(row[1:], dtype=float)
                            # pandas and Python can round the last decimal differently.
                            if not np.allclose(values, X.loc[row[0]].to_numpy(), rtol=1e-14, atol=1e-14):
                                raise ValueError(f"Expression mismatch in {row[0]}")
                    else:
                        for row in reader:
                            if row[0] in labels:
                                raise ValueError("Duplicate identifier in original labels.")
                            labels[row[0]] = row[1]
    if original_ids != X.index.tolist() or labels != y.to_dict():
        raise ValueError("Original sample order or labels differ from prepared data.")


def validate_data(full_archive=False):
    """Validate source hashes and save a separate derived audit report."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    pairs = {"UCI_original_download.zip": "original_archive_sha256",
             "third_party_gene_mapping.csv": "mapping_sha256",
             "ONCOSENTRY_gene_named_801x20531.csv.gz": "mapped_dataset_sha256"}
    hashes = {name: sha256(DATA_DIR / name) for name in pairs}
    for name, key in pairs.items():
        if hashes[name] != manifest[key]:
            raise ValueError(f"File hash changed: {name}")
    X, y, cross = load_dataset()
    if full_archive:
        validate_original_archive(X, y)
    report = {"validated_at_utc": utc_now(), "samples": len(X), "features": X.shape[1],
              "class_counts": y.value_counts().to_dict(), "missing_values": 0,
              "duplicate_ids": 0, "duplicate_expression_rows": 0,
              "unresolved_symbols": cross.loc[cross.gene_symbol.eq(""), "uci_feature"].tolist(),
              "source_sha256": hashes, "original_archive_compared": full_archive,
              "mapping_quality": manifest["mapping_quality"], "passed": True}
    write_json(RESULT_DIR / "data_validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-archive", action="store_true")
    print(json.dumps(validate_data(parser.parse_args().full_archive), indent=2))

