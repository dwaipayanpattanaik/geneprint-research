"""Auditable candidate-gene annotations and optional training-only stability.

The third-party crosswalk is provisional. A citation supporting the biology of a
named gene does not independently verify that a UCI feature maps to that gene.
No annotation download, enrichment claim, or causal biomarker discovery is made.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from config import MODEL_DIR, RESULT_DIR, SEED

# Curated records are added only after reviewing an authoritative source for an
# actually selected candidate gene. Keys include Entrez IDs to avoid symbol-only
# ambiguity. The cited biology is separate from our statistical associations.
CURATED_BIOLOGY: dict[tuple[str, str], dict[str, str]] = {}


def annotate_selected_genes() -> pd.DataFrame:
    """Join saved feature identities to explicitly curated, cited descriptions."""
    registry = json.loads((MODEL_DIR / "registry.json").read_text())
    frames = []
    for manifest in registry["models"]:
        if not manifest["model_id"].startswith("compact_"):
            continue
        path = MODEL_DIR / f"{manifest['model_id']}_features.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"entrez_id": str}).fillna("")
            frame["model_id"] = manifest["model_id"]
            frames.append(frame)
    if not frames:
        raise RuntimeError("Train compact models before annotating selected genes.")
    selected = pd.concat(frames, ignore_index=True)
    membership = selected.groupby("uci_feature")["model_id"].agg(lambda names: ";".join(sorted(set(names))))
    selected = selected.drop_duplicates("uci_feature")
    rows = []
    for _, feature in selected.iterrows():
        symbol, entrez = str(feature["gene_symbol"]), str(feature["entrez_id"])
        evidence = CURATED_BIOLOGY.get((symbol, entrez), {})
        rows.append({
            "uci_feature": feature["uci_feature"], "gene_symbol": symbol,
            "entrez_id": entrez, "mapping_status": feature["mapping_status"],
            "models": membership[feature["uci_feature"]],
            "biological_function": evidence.get("biological_function", "Function not curated in this project."),
            "interpretation": evidence.get("interpretation", "Selected by a statistical association with training labels; biological explanation not established."),
            "source_url": evidence.get("source_url", ""),
            "gene_record_url": f"https://www.ncbi.nlm.nih.gov/gene/{entrez}" if entrez else "",
            "review_status": "Source reviewed; feature mapping remains provisional" if evidence else "Biological function not reviewed",
            "mapping_caution": "A gene-level citation does not independently verify the original UCI-to-TCGA feature order.",
        })
    output = RESULT_DIR / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    annotations = pd.DataFrame(rows)
    annotations.to_csv(output / "biological_annotations.csv", index=False)
    reviewed = annotations[annotations["source_url"] != ""]
    lines = ["# Exploratory biological interpretation", "",
             "Candidate symbols are provisional third-party annotations. These descriptions concern the named gene; they do not prove that the UCI feature was mapped correctly.", "",
             "Feature-selection scores measure associations with training labels. They do not identify causal cancer mechanisms or establish new biomarkers. Tissue identity is a plausible alternative explanation that this experiment cannot separate from malignancy.", "",
             f"Selected original features: {len(annotations)}. Source-reviewed candidate descriptions: {len(reviewed)}.", ""]
    for _, row in reviewed.iterrows():
        lines += [f"## {row['gene_symbol']} — {row['uci_feature']} (Entrez {row['entrez_id']})", "",
                  row["biological_function"], "", row["interpretation"], "",
                  f"Source: {row['source_url']}", ""]
    lines += ["## Scope", "", "No independent TCGA index verification or pathway enrichment has been completed. Genes without a reviewed source retain identifiers and are explicitly marked as unreviewed.", ""]
    (output / "biological_interpretation.md").write_text("\n".join(lines))
    return annotations


def run_feature_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit imputation and feature selection in three training-only folds.

    Jaccard(A, B) = |A intersection B| / |A union B|. This examines selection
    stability within the discovery data and never uses validation/test labels.
    Three folds are exploratory and do not establish external reproducibility.
    """
    from sklearn.feature_selection import SelectKBest
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from src.preprocess import safe_f_classif
    from src.validate_data import load_dataset

    splits = json.loads((RESULT_DIR / "split_manifest.json").read_text())["splits"]
    X, y, crosswalk = load_dataset()
    X_train = X.loc[splits["train"]]
    y_train = y.loc[X_train.index]
    selected_sets = {}
    selections = []
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (fit_indices, _) in enumerate(folds.split(X_train, y_train), start=1):
        for k in (10, 25):
            pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                                 ("selector", SelectKBest(safe_f_classif, k=k))])
            pipeline.fit(X_train.iloc[fit_indices].to_numpy(), y_train.iloc[fit_indices].to_numpy())
            selector = pipeline.named_steps["selector"]
            indices = selector.get_support(indices=True)
            features = X_train.columns[indices]
            selected_sets[k, fold] = set(features)
            for feature, index in zip(features, indices):
                selections.append({"feature_count": k, "fold": fold, "uci_feature": feature,
                                   "selection_score": float(selector.scores_[index]),
                                   "fit_sample_count": len(fit_indices), "split": "train"})
    overlap = []
    for k in (10, 25):
        for fold_a, fold_b in itertools.combinations((1, 2, 3), 2):
            a, b = selected_sets[k, fold_a], selected_sets[k, fold_b]
            overlap.append({"feature_count": k, "fold_a": fold_a, "fold_b": fold_b,
                            "intersection": len(a & b), "union": len(a | b),
                            "jaccard": len(a & b) / len(a | b), "split": "train"})
    selections = pd.DataFrame(selections)
    overlaps = pd.DataFrame(overlap)
    frequency = selections.groupby(["feature_count", "uci_feature"]).size().reset_index(name="selected_folds")
    frequency["fold_count"] = 3
    frequency["selection_frequency"] = frequency["selected_folds"] / 3
    annotations = crosswalk.copy()
    if "uci_feature" not in annotations.columns:
        annotations = annotations.reset_index()
    frequency = frequency.merge(annotations[["uci_feature", "gene_symbol", "entrez_id", "mapping_status"]],
                                on="uci_feature", how="left", validate="many_to_one")
    output = RESULT_DIR / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output / "stability_selections.csv", index=False)
    overlaps.to_csv(output / "feature_stability.csv", index=False)
    frequency.to_csv(output / "feature_selection_frequency.csv", index=False)
    return selections, overlaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability", action="store_true", help="Also refit selection within three training-only folds.")
    args = parser.parse_args()
    annotations = annotate_selected_genes()
    print(f"Wrote {len(annotations)} feature annotations; {(annotations.source_url != '').sum()} source-reviewed descriptions.")
    if args.stability:
        _, overlap = run_feature_stability()
        print(overlap.groupby("feature_count")["jaccard"].mean().to_string())
