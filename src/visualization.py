"""Export publication-ready figures together with their numerical source tables.

Every figure reads completed experiment artifacts. Regenerating figures does
not fit models or change the held-out evaluation protocol.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from config import CLASS_NAMES, CLASSES, MODEL_DIR, RESULT_DIR, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NAVY = "#0B1F3A"
GOLD = "#D4AF37"
CLASS_COLORS = {"BRCA": "#B84A7B", "COAD": "#C17C18", "KIRC": "#238C75",
                "LUAD": "#3C79B7", "PRAD": "#8057A6"}
MODEL_COLORS = ["#0B1F3A", "#238C75", "#B58B12", "#8057A6", "#B84A7B"]
CHART_DIR = RESULT_DIR / "charts"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY,
                         "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF"})


def _save(fig: plt.Figure, name: str, source: pd.DataFrame, note: str) -> list[str]:
    """Save PNG/SVG and a CSV with the exact numbers used by a figure."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.text(0.02, 0.012, note, fontsize=8, color="#46566C", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    outputs = []
    for extension in ("png", "svg"):
        path = CHART_DIR / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(str(path.relative_to(ROOT)))
    source.to_csv(CHART_DIR / f"{name}_source.csv", index=False)
    plt.close(fig)
    return outputs


def _count_note(frame: pd.DataFrame, split: str) -> str:
    sizes = sorted(frame["n_samples"].dropna().astype(int).unique())
    return f"Evaluation: {split}; n={','.join(map(str, sizes))} tumor samples. Scores are research measurements."


def class_distribution() -> list[str]:
    """Plot the full public dataset's label composition, not model performance."""
    from src.validate_data import load_dataset

    _, labels, _ = load_dataset()
    source = labels.value_counts().reindex(CLASSES).rename_axis("class").reset_index(name="samples")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(source["class"], source["samples"], color=[CLASS_COLORS[c] for c in source["class"]])
    ax.bar_label(bars, padding=4)
    ax.set(title="Public tumor dataset: class distribution", xlabel="Tumor category", ylabel="Sample count")
    ax.set_ylim(0, source["samples"].max() * 1.16)
    return _save(fig, "class_distribution", source,
                 f"All {len(labels)} public samples; five tumor categories; no healthy controls.")


def feature_count_figures(summary: pd.DataFrame) -> list[str]:
    """Show feature-count comparisons separately for validation and final test."""
    outputs = []
    for split, frame in summary.groupby("split", sort=False):
        learned = frame[frame["model_id"] != "majority"].sort_values("n_features")
        baseline = frame[frame["model_id"] == "majority"]
        for metric, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
                              ("balanced_accuracy", "Balanced accuracy")):
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(learned["n_features"], learned[metric], marker="o", color=NAVY,
                    linewidth=2, markersize=7, label="Logistic regression")
            if len(baseline):
                ax.axhline(float(baseline.iloc[0][metric]), color=GOLD, linestyle="--",
                           linewidth=2, label="Majority-class baseline")
            ax.set_xscale("log")
            ax.set_xticks(learned["n_features"].unique(), [f"{int(k):,}" for k in learned["n_features"].unique()])
            ax.set_ylim(0, 1.04)
            ax.set(title=f"{title} versus gene-feature count — {split}",
                   xlabel="Gene-expression features (logarithmic axis)", ylabel=title)
            ax.legend(loc="lower right")
            outputs += _save(fig, f"feature_count_{metric}_{split}", frame,
                             _count_note(frame, split) + " Test results did not guide model selection.")
    return outputs


def per_class_figures(per_class: pd.DataFrame, model_id: str) -> list[str]:
    """Display per-class scores and one-versus-rest error counts."""
    outputs = []
    for split in ("validation", "test"):
        source = per_class[(per_class["model_id"] == model_id) & (per_class["split"] == split)]
        if source.empty:
            continue
        source = source.set_index("class").reindex(CLASSES).reset_index()
        x = np.arange(len(CLASSES))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for index, (metric, color) in enumerate(zip(("precision", "recall", "f1"), (NAVY, GOLD, "#238C75"))):
            ax.bar(x + (index - 1) * 0.24, source[metric], width=0.24, label=metric.title(), color=color)
        ax.set_xticks(x, [f"{c}\n(n={int(n)})" for c, n in zip(CLASSES, source["support"])])
        ax.set_ylim(0, 1.12)
        ax.set(title=f"Per-class performance — {model_id}, {split}", ylabel="Score", xlabel="Tumor category (true-class support)")
        ax.legend(loc="lower right")
        note = f"{split.title()} set; n={int(source['support'].sum())}. Each class is evaluated against all other classes."
        outputs += _save(fig, f"per_class_{model_id}_{split}", source, note)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for offset, metric, label, color in ((-0.18, "false_positives", "False positives", GOLD),
                                             (0.18, "false_negatives", "False negatives", NAVY)):
            bars = ax.bar(x + offset, source[metric], width=0.36, label=label, color=color)
            ax.bar_label(bars, padding=3, fmt="%d")
        ax.set_xticks(x, CLASSES)
        maximum = float(source[["false_positives", "false_negatives"]].max().max())
        ax.set_ylim(0, max(1.5, maximum * 1.25 + 0.5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set(title=f"Actual classification errors — {model_id}, {split}", xlabel="Tumor category", ylabel="Sample count")
        ax.legend()
        outputs += _save(fig, f"error_counts_{model_id}_{split}", source, note)
    return outputs


def confusion_figures() -> list[str]:
    """Render all completed confusion matrices, keeping class order explicit."""
    path = RESULT_DIR / "metrics" / "confusion_matrices.json"
    if not path.exists():
        return []
    matrices = json.loads(path.read_text())
    outputs = []
    for model_id, splits in matrices.items():
        for split, content in splits.items():
            matrix = np.asarray(content["matrix"], dtype=int)
            labels = content["labels"]
            fig, ax = plt.subplots(figsize=(6.9, 5.6))
            sns.heatmap(matrix, annot=True, fmt="d", cmap=sns.light_palette(NAVY, as_cmap=True),
                        xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Sample count"})
            ax.set(title=f"Confusion matrix — {model_id}, {split}", xlabel="Predicted tumor category", ylabel="True tumor category")
            source = pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("true_label").reset_index()
            outputs += _save(fig, f"confusion_{model_id}_{split}", source,
                             f"{split.title()} set; n={matrix.sum()}. Diagonal entries are correct predictions.")
    return outputs


def robustness_figures() -> list[str]:
    """Plot mean and min–max across seeds; the band is not a confidence interval."""
    path = RESULT_DIR / "experiments" / "robustness.csv"
    if not path.exists():
        return []
    all_results = pd.read_csv(path)
    outputs = []
    for experiment, source in all_results.groupby("experiment"):
        for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
            fig, ax = plt.subplots(figsize=(8.8, 5.1))
            for color, (model_id, group) in zip(MODEL_COLORS, source.groupby("model_id")):
                aggregate = group.groupby("level")[metric].agg(["mean", "min", "max"]).sort_index()
                x = aggregate.index.to_numpy(dtype=float)
                if experiment == "missingness":
                    x = x * 100
                ax.plot(x, aggregate["mean"].to_numpy(), marker="o", label=model_id, color=color, linewidth=2)
                ax.fill_between(x, aggregate["min"].to_numpy(), aggregate["max"].to_numpy(), color=color, alpha=0.12)
            ax.set_ylim(0, 1.04)
            ax.set(title=f"Synthetic {experiment} sensitivity — locked test set", ylabel=label,
                   xlabel=("Noise standard deviation / training gene standard deviation"
                           if experiment == "noise" else "Expected missing measurements (%)"))
            ax.legend(loc="lower left")
            outputs += _save(fig, f"robustness_{experiment}_{metric}", source,
                             f"n={int(source['n_samples'].iloc[0])}; 3 perturbation seeds. Band: seed min–max, not a confidence interval. Synthetic errors are not clinical validation.")
    return outputs


def sample_scores(model_id: str) -> list[str]:
    """Plot actual saved class probabilities for one public held-out sample."""
    path = RESULT_DIR / "predictions" / "test_predictions.csv"
    if not path.exists():
        return []
    predictions = pd.read_csv(path)
    candidates = predictions[predictions["model_id"] == model_id]
    if candidates.empty:
        return []
    row = candidates.iloc[0]
    source = pd.DataFrame({"class": CLASSES, "model_score": [row[f"score_{c}"] for c in CLASSES]})
    source["sample_id"] = row["sample_id"]
    source["model_id"] = model_id
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(source["class"], source["model_score"] * 100, color=[CLASS_COLORS[c] for c in CLASSES])
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlim(0, 110)
    ax.set(title=f"Model-estimated class scores — {row['sample_id']}", xlabel="Model score (%)", ylabel="Tumor category")
    return _save(fig, "example_class_scores", source,
                 f"Locked test sample; model={model_id}; true label={row['true_label']}. Scores are not a person's probability of cancer.")


def gene_expression_figures() -> list[str]:
    """Use training-only expression distributions for ten selected features."""
    path = RESULT_DIR / "experiments" / "gene_expression.csv"
    features_path = MODEL_DIR / "compact_10_features.csv"
    if not path.exists() or not features_path.exists():
        return []
    expression = pd.read_csv(path)
    features = pd.read_csv(features_path).fillna("")
    source = expression[(expression["split"] == "train") & expression["uci_feature"].isin(features["uci_feature"])]
    if source.empty:
        return []
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.2))
    for ax, (_, feature) in zip(axes.flat, features.iterrows()):
        gene = feature["uci_feature"]
        subset = source[source["uci_feature"] == gene]
        sns.boxplot(data=subset, x="cancer_type", y="value", order=CLASSES, hue="cancer_type",
                    hue_order=CLASSES, palette=CLASS_COLORS, legend=False, ax=ax, fliersize=1.5)
        ax.set(title=f"{feature['gene_symbol'] or 'Unresolved'}\n{gene}", xlabel="", ylabel="Supplied expression scale")
        ax.tick_params(axis="x", labelsize=8, rotation=35)
    fig.suptitle("Ten selected features: training-set expression by tumor category", fontsize=15)
    return _save(fig, "selected_gene_expression", source,
                 f"Training samples only (n={source['sample_id'].nunique()}); candidate symbols remain provisional. Association does not establish causation.")


def error_examples(model_id: str) -> list[str]:
    """Show genuine errors; use explicitly labeled perturbations if clean errors are absent."""
    path = RESULT_DIR / "predictions" / "test_predictions.csv"
    if not path.exists():
        return []
    predictions = pd.read_csv(path)
    source = predictions[(predictions["model_id"] == model_id) &
                         (predictions["true_label"] != predictions["predicted_label"])].copy()
    label = "Unperturbed locked test set"
    columns = ["sample_id", "true_label", "predicted_label", "max_score"]
    if source.empty:
        robust_path = RESULT_DIR / "experiments" / "robustness_errors.csv"
        if robust_path.exists():
            robust = pd.read_csv(robust_path)
            source = robust[(robust["model_id"] == model_id) & (robust["level"] > 0)].copy()
            source = source.sort_values(["level", "seed"]).drop_duplicates("sample_id")
            columns += ["experiment", "level"]
            label = "Synthetic perturbation errors; no unperturbed test errors for this model"
    if source.empty:
        return []
    source = source.head(8)
    display = source[columns].copy()
    display["max_score"] = display["max_score"].map(lambda score: f"{score:.3f}")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F7F7F2" if row % 2 else "white")
    ax.set_title(f"Observed failure examples — {model_id}", pad=14)
    return _save(fig, "genuine_error_examples", source,
                 label + ". The cause of an individual error is unknown; scores are uncalibrated model estimates.")


def generate_charts() -> list[str]:
    """Create available figures and a manifest; missing experiments are skipped."""
    _style()
    summary = pd.read_csv(RESULT_DIR / "metrics" / "summary.csv")
    registry = json.loads((MODEL_DIR / "registry.json").read_text())
    selected = registry["selected_model"]
    outputs = class_distribution()
    outputs += feature_count_figures(summary)
    outputs += per_class_figures(pd.read_csv(RESULT_DIR / "metrics" / "per_class.csv"), selected)
    outputs += confusion_figures()
    outputs += robustness_figures()
    outputs += sample_scores(selected)
    outputs += gene_expression_figures()
    outputs += error_examples(selected)
    (CHART_DIR / "figure_manifest.json").write_text(json.dumps({"files": outputs,
        "source_tables": "Each figure has a matching *_source.csv.",
        "integrity": "Figures are generated from measured results; empty or missing experiments are not invented."}, indent=2) + "\n")
    return outputs


if __name__ == "__main__":
    print("\n".join(generate_charts()))
