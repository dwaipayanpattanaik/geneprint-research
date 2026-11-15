"""Build research reports from saved experimental outputs, never invented metrics.

Run ``python -m src.reporting`` after training and evaluation. Missing artifacts
are described as incomplete rather than filled with demonstration numbers.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read an optional CSV artifact; absent outputs produce an empty list."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    """Read an optional JSON object; missing files are not experimental results."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def number(value: Any, decimals: int = 4) -> str:
    """Format a measured number, retaining an explicit missing-data marker."""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "not available"


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a small Markdown table without a separate formatting dependency."""
    if not rows:
        return "**Incomplete: no corresponding experimental output is available.**"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [str(row.get(key, "not available")).replace("|", "\\|").replace("\n", " ") for key, _ in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def performance_table(rows: list[dict[str, str]]) -> str:
    """Present each model's measured scores, keeping evaluation splits separate."""
    formatted = [{**row, **{key: number(row.get(key)) for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}} for row in rows]
    return table(formatted, [("model_id", "Model"), ("n_features", "Features"), ("n_samples", "N"), ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted F1")])


def summarize_robustness(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Average repeated perturbation seeds; seed variation is not a cohort CI."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["model_id"], row["experiment"], row["level"]), []).append(row)
    result = []
    for (model, experiment, level), group in sorted(groups.items()):
        scores = [float(row["macro_f1"]) for row in group]
        accuracy = [float(row["accuracy"]) for row in group]
        result.append({"model_id": model, "experiment": experiment, "level": level, "seeds": len(group), "accuracy": number(sum(accuracy) / len(accuracy)), "macro_f1": number(sum(scores) / len(scores)), "macro_f1_min": number(min(scores)), "macro_f1_max": number(max(scores))})
    return result


def error_table(rows: list[dict[str, str]], limit: int = 10) -> str:
    """Show genuine saved mistakes; a perfect finite test set is reported plainly."""
    if not rows:
        return "**Incomplete: prediction artifacts are not available.**"
    errors = [row for row in rows if row.get("true_label") != row.get("predicted_label")]
    if not errors:
        return "No mistakes were observed in these saved clean predictions. This finite test result does not establish perfect accuracy in new cohorts; separately labeled perturbation failures are examined when available."
    formatted = [{**r, "max_score": number(r.get("max_score"))} for r in errors[:limit]]
    return table(formatted, [("model_id", "Model"), ("sample_id", "Public sample"), ("true_label", "True category"), ("predicted_label", "Predicted category"), ("max_score", "Top model score")])


def generate_reports(root: Path = ROOT) -> list[Path]:
    """Write the scientific report, hackathon write-up and measured README block.

    Inputs are saved training/evaluation files beneath ``root``. The function
    returns paths written and deliberately never trains or alters a model.
    """
    result_dir = root / "results"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(result_dir / "metrics" / "summary.csv")
    per_class = read_csv(result_dir / "metrics" / "per_class.csv")
    validation = [row for row in summary if row.get("split") == "validation"]
    test = [row for row in summary if row.get("split") == "test"]
    predictions = read_csv(result_dir / "predictions" / "test_predictions.csv")
    perturbations = read_csv(result_dir / "experiments" / "robustness.csv")
    registry = read_json(root / "models" / "registry.json")
    manifest = read_json(root / "oncosentry_data" / "manifest.json")
    split = read_json(result_dir / "split_manifest.json")
    if not split:
        split = read_json(result_dir / "experiments" / "split_manifest.json")
    split_counts = {key: len(value) for key, value in split.get("splits", {}).items()}
    selected = registry.get("selected_model", "not yet selected")
    manual = registry.get("manual_model", "compact_10")
    chosen = next((row for row in test if row["model_id"] == selected), None)
    manual_result = next((row for row in test if row["model_id"] == manual), None)
    baseline = next((row for row in test if row["model_id"] in ("majority", "baseline", "dummy", "dummy_majority")), None)
    if baseline is None:
        baseline = next((row for row in test if "majority" in row["model_id"] or "dummy" in row["model_id"]), None)
    selected_class = [row for row in per_class if row.get("split") == "test" and row["model_id"] == selected]
    class_table = table([{**r, **{k: number(r.get(k)) for k in ("precision", "recall", "f1")}} for r in selected_class], [("class", "Class"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("support", "Support"), ("false_positives", "FP"), ("false_negatives", "FN")])
    robust_table = table(summarize_robustness(perturbations), [("model_id", "Model"), ("experiment", "Experiment"), ("level", "Level"), ("seeds", "Seeds"), ("accuracy", "Mean accuracy"), ("macro_f1", "Mean macro-F1"), ("macro_f1_min", "Minimum F1"), ("macro_f1_max", "Maximum F1")])
    features = read_csv(root / "models" / f"{manual}_features.csv")
    gene_table = table(features, [("uci_feature", "Original feature"), ("gene_symbol", "Candidate symbol"), ("entrez_id", "Entrez ID"), ("selection_score", "Training ANOVA score"), ("mapping_status", "Mapping status")])
    selected_errors = [row for row in predictions if row.get("model_id") == selected]
    errors = [row for row in selected_errors if row["true_label"] != row["predicted_label"]]
    fp = sum(int(float(row["false_positives"])) for row in selected_class) if selected_class else None
    fn = sum(int(float(row["false_negatives"])) for row in selected_class) if selected_class else None
    measured_sentence = (f"The validation-selected model `{selected}` achieved held-out accuracy {number(chosen['accuracy'])}, balanced accuracy {number(chosen['balanced_accuracy'])}, and macro-F1 {number(chosen['macro_f1'])} on {chosen['n_samples']} test samples." if chosen else "Final held-out metrics are incomplete; no performance claim is made.")
    baseline_sentence = (f"The majority-class baseline achieved accuracy {number(baseline['accuracy'])} and macro-F1 {number(baseline['macro_f1'])}." if baseline else "The majority-class baseline result is not yet available.")
    manual_sentence = (f"The ten-feature model achieved accuracy {number(manual_result['accuracy'])} and macro-F1 {number(manual_result['macro_f1'])} on the same held-out set." if manual_result else "The ten-feature model's final result is not yet available.")
    conclusion = "The saved tables support a comparison of compact and full-feature models within this dataset; external validity remains untested."
    full = next((r for r in test if int(float(r.get("n_features", "0"))) == 20531 and r is not baseline), None)
    if full and manual_result:
        delta = float(manual_result["macro_f1"]) - float(full["macro_f1"])
        conclusion = f"The ten-feature model's test macro-F1 differed from the full-feature model by {delta:+.4f}. This is a descriptive comparison on one fixed split, not a proof of equivalence or a significance test. Compact signatures use fewer measured inputs, but cost savings and clinical usefulness were not evaluated."
    scientific = f"""# ONCOSENTRY: Cancer-Type Classification Using Compact, Biologically Annotated Gene-Expression Signatures and Measurement-Robustness Testing

Research interface: **Cancer Gene Lab**. Team: Dwaipayan, Jonathan and Vedanth. Veritas AI International High School Hackathon 2026; AI + Health — Your Own Idea.

**This project is a prototype and is not medical advice.**

## Abstract

We investigate whether supervised feature selection can produce a compact expression signature that retains tumor-category classification performance, and how simulated measurement perturbations affect predictions. The existing UCI dataset contains 801 tumor samples and 20,531 expression features across five categories. A majority-class baseline, regularized logistic regression, and selected-feature models are evaluated with a reproducible training/validation/test split. {measured_sentence} {baseline_sentence} {manual_sentence} Gene names remain provisional because the third-party mapping has not been independently verified end-to-end against original ordered TCGA annotation. The system does not distinguish healthy people from cancer patients.

## Introduction and biological background

Gene expression reflects the production of RNA from genetic information. Different cell types and conditions can exhibit different expression patterns. RNA sequencing measures RNA-related signals; a measured expression value is not a direct measurement of a gene's causal contribution to cancer. These concepts are described by [NHGRI](https://www.genome.gov/genetics-glossary/Gene-Expression). Tissue-of-origin differences are a plausible source of class separation in this dataset. They must not be interpreted automatically as new cancer mechanisms.

## Research problem, question and hypotheses

The intended user is a molecular pathology research assistant or oncology laboratory researcher comparing public tumor samples and investigating which inputs affect classification reliability. Can a compact, biologically annotated signature preserve performance while remaining reliable under simulated measurement noise compared with a full-feature classifier?

1. A selected subset may retain much of the full-feature model's performance.
2. Feature count may influence sensitivity to noise or missing measurements.
3. Some selected features may have established tissue-related biology; this requires source-supported interpretation and does not follow from predictive selection alone.

The application supports exploratory research, teaching and reproducible comparison. It does not estimate cancer risk, stage, survival, treatment response or appropriate treatment.

## Dataset provenance and integrity

[UCI Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq), DOI [10.24432/C5R88H](https://doi.org/10.24432/C5R88H), is a public TCGA/PANCAN-derived dataset. UCI describes feature order as consistent with its original submission. The prepared inputs were reused; the original archive was not redownloaded or replaced.

Local validation checks 801 sample identifiers, 20,531 ordered features, finite values, duplicate identifiers, numeric duplicate expression rows, original archive label alignment and file hashes. The initial read-only audit found no missing/nonfinite values or duplicate expression rows, and all 16,445,331 saved measurements matched the original archive. Class counts are BRCA 300, COAD 78, KIRC 146, LUAD 141 and PRAD 136. A clean local audit does not establish patient-level independence or absence of hidden biological confounding.

## Biological feature mapping

The saved crosswalk retains original `gene_N`, candidate symbol, Entrez ID and original third-party TCGA-style string. The mapping source is [Roman-Young's gene_mapping.csv](https://github.com/Roman-Young/tcga-pancan-tumor-classifier/blob/main/gene_mapping.csv). Structural consistency was checked across all 20,531 entries. Exactly 29 symbols are unresolved, corresponding to `gene_0` through `gene_28`; they retain anonymous feature identities. The symbol SLC35E2 occurs for two distinct entries. Original feature IDs, rather than symbols alone, are model keys.

**Mapping status: provisional.** Agreement with the saved third-party table is not independent verification of the biological identity assigned to each original UCI feature. No additional mapping dataset was downloaded for this build.

## Preprocessing, split and model architecture

Sample IDs and target labels are excluded from model features. The split manifest records disjoint training, validation and test sample IDs using seed 42 and stratification. Recorded split sizes: `{json.dumps(split_counts, sort_keys=True) if split_counts else 'incomplete'}`. Training fits the preprocessing, including median imputation and scaling, and supervised ANOVA feature selection where applicable. Validation guides model selection. The held-out test is evaluated after model and evaluation choices are locked. See [METHODS.md](METHODS.md) and the saved protocol/registry for exact implementation details.

The majority-class baseline uses `DummyClassifier(strategy="most_frequent")`. Regularized logistic regression is evaluated using all 20,531 features and selected signatures of 10, 25, 50, 100 and 500 features. The pipeline applies the same fitted preprocessing at prediction time. Feature selection uses training labels only; annotation is metadata, not a selection shortcut. No neural-network or GPU requirement is introduced.

## Evaluation metrics

For class c, TP is a correct prediction of c; FP is a different true class predicted as c; FN is a true c predicted as another class. Precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2 precision recall/(precision+recall). Zero denominators are handled explicitly by the evaluation code. Support is the number of true samples in a class. Balanced accuracy averages class recall. Macro-F1 gives each class equal weight; weighted F1 weights by support. Accuracy is correct predictions divided by all evaluated samples.

In single-label multiclass evaluation, every wrong prediction produces one FP for the predicted class and one FN for the true class. Summing FP or FN therefore counts errors; adding the two would count each error twice. Confusion-matrix rows are true labels and columns are predictions. The class order is stored with each matrix.

## Validation results and selection

{performance_table(validation)}

The saved registry selects `{selected}`. Selection must be based on validation results and the recorded tie-break rule, not final test performance. Validation and test tables are intentionally separate.

## Final held-out results

{performance_table(test)}

{measured_sentence} {baseline_sentence}

Per-class results for the selected model:

{class_table}

Aggregated one-versus-rest FP = {fp if fp is not None else 'incomplete'}; FN = {fn if fn is not None else 'incomplete'}. These are tumor-category errors, not counts of healthy patients incorrectly diagnosed with cancer.

## Feature-count analysis

{conclusion}

The comparison holds the dataset split fixed. Selected features can be correlated and feature rankings can be unstable. ANOVA scoring is univariate; it does not measure a gene's causal effect or guarantee that the selected set is uniquely optimal. Feature-count graphs use the saved metrics and identify the evaluation split. No equivalence margin or independent replication cohort was specified, so claims of statistical equivalence are not warranted.

## Noise and missingness experiments

{robust_table}

The aggregate rows above summarize repeated seeds at each saved perturbation level. The experiment code defines Gaussian noise relative to training-derived feature standard deviations and simulated missing entries imputed by training-fitted statistics. A level of zero is the clean reference. Seed-to-seed ranges reflect random perturbations of the same samples, not confidence intervals across independent patient cohorts. Synthetic perturbations are controlled computational stress tests; they do not reproduce all laboratory errors, platform differences, tumor purity effects or batch effects. See the saved experiment protocol for actual seeds, levels and evaluated split.

## Ten-feature signature and biological interpretation

{gene_table}

The selected features above come from the saved model, not from prechosen example genes. Training expression distributions may suggest which categories differ, but these associations do not establish mechanisms. Source-backed gene notes, when completed, are provided in [BIOLOGICAL_INTERPRETATION.md](BIOLOGICAL_INTERPRETATION.md). Identifier correctness in NCBI cannot on its own validate the crosswalk's original feature order. Pathway enrichment and biomarker-discovery claims are outside the completed core experiment unless a separately documented analysis exists.

## Genuine error analysis

{error_table(selected_errors)}

Across all saved clean test predictions, including the baseline and compact alternatives:

{error_table(predictions)}

For a baseline error, the demonstrated cause is its rule of always choosing the training majority class. For learned-model errors, similar expression patterns, measurement variability or limited training coverage are hypotheses, not demonstrated causes. Public sample IDs here are anonymous dataset row identifiers. See saved prediction CSVs for every example and every class score. Separately generated robustness-error artifacts, when present, label synthetic conditions explicitly.

## Uncertainty and calibration

Class scores are returned by the persisted classifier's `predict_proba` and interpreted only among the five represented categories. Their sum is checked numerically. They are not medically validated probabilities. A validation-derived review threshold is recorded per model; a review flag is a research-interface convention, not a clinical recommendation. Validation reliability summaries, when available, are exploratory because the sample is small. High accuracy alone does not establish probability calibration. [Scikit-learn's calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) explains this distinction.

Training-range checks identify individual feature values outside observed training minima/maxima. Values inside those ranges can still be unrealistic jointly or outside the training population; the range flag is not a validated out-of-distribution detector. No calibration correction is claimed unless explicitly fitted and recorded in the saved model.

## Manual inference and application

The ten-gene page uses the model's exact selected schema and fitted compact pipeline. A public example can prefill the fields; users can edit measurements or supply a one-row CSV. Feature IDs, numeric values, finite values and missing entries are validated. Columns are matched by unique identifiers before inference. Automated parity checks compare compact input predictions with the full research pipeline. The app loads persisted artifacts and does not retrain on refresh. It does not collect patient names, symptoms or medical records. Arbitrary edited values may not represent a biologically realistic sample.

## Limitations and ethics

Only five tumor classes are represented; there are no healthy controls. A random within-dataset split cannot establish transportability across hospitals, platforms, demographics or unseen cancer types. Cohort metadata for patient grouping and subgroup fairness assessment are unavailable in these prepared inputs. Hidden batch effects or patient overlap cannot be ruled out from anonymous row IDs alone. There are many more candidate features than samples. Selected biological names remain provisional. Synthetic robustness and finite held-out performance do not establish clinical safety. No wet-lab validation, clinical trial, treatment analysis or cost study was performed.

Use public research examples or invented values only. Do not enter personal medical information. Reduced feature count may motivate future assay research, but does not by itself demonstrate cheaper clinical testing or remove the need for laboratory equipment. See [LIMITATIONS.md](LIMITATIONS.md) for the full scope.

## Conclusion and future work

{conclusion} {manual_sentence} The experiments address internal dataset classification and computational robustness. Independent annotation-order verification, patient-aware external validation, repeated training-only selection stability, appropriately powered calibration analysis and measurement-platform compatibility studies would strengthen subsequent work. Any new biological reference download requires separate permission.

## Reproducibility

Use the pinned environment and commands in [README.md](../README.md). Keep the original prepared files unchanged. Preserve split manifests, model manifests, selected feature files, version records, numerical chart sources and measured prediction outputs. Running the report generator updates prose from saved results without fitting or selecting models. Publishing and public deployment have not been authorized; proposed URLs are not verified deployments.

## AI coding-assistance disclosure and team contributions

Codex generated and edited software, documentation and tests and assisted with experiment execution under the user's direction. ChatGPT/Codex assistance must be disclosed to judges. Dwaipayan coordinated this request; Jonathan and Vedanth are named teammates. Their proposed responsibilities are not presented as verified completed work. See [TEAM_CONTRIBUTIONS.md](TEAM_CONTRIBUTIONS.md) and update it only with actual reviewed contributions.

## References

See [REFERENCES.md](REFERENCES.md), the dataset manifest, and links above. Sources distinguish dataset provenance, provisional mapping, established biological descriptions and software methods.
"""
    writeup = f"""# ONCOSENTRY — Cancer Gene Lab

Track: AI + Health — Your Own Idea  
Team: Dwaipayan, Jonathan and Vedanth

## The problem

A molecular pathology research assistant needs reproducible ways to compare tumor-expression patterns and investigate whether a small feature signature is reliable. Our research asks whether compact signatures can preserve classification performance while tolerating simulated measurement errors.

## What we built

Cancer Gene Lab is a Streamlit research application. Users select a public tumor sample or enter the exact ten selected expression features, run a saved classifier, inspect five class scores and uncertainty, and explore measured model comparisons and perturbation experiments. Inputs use the supplied research-data scale.

## What we added

**Our own idea**: compare compact, provisionally annotated gene signatures with a full-feature model and a majority baseline, then test sensitivity to controlled noise and missing measurements. The contribution is the reproducible comparison and usable research workflow; no novel cancer biomarker is claimed.

## How it works

The existing UCI dataset contains 801 tumors, 20,531 features and five categories. A stratified split separates training, validation and testing. Training-only median imputation, ANOVA feature selection and scaling feed regularized logistic regression. Validation determines selection; final test results are reported separately. Saved pipelines power real inference, including a ten-input pipeline checked against the research model.

## Results

{measured_sentence} {baseline_sentence} {manual_sentence} Selected-model test FP total: {fp if fp is not None else 'incomplete'}; FN total: {fn if fn is not None else 'incomplete'} (one-versus-rest totals; do not add them together).

{('A genuine clean test error is ' + errors[0]['sample_id'] + ': true ' + errors[0]['true_label'] + ', predicted ' + errors[0]['predicted_label'] + ', score ' + number(errors[0].get('max_score')) + '.') if errors else 'No clean test error is claimed for the selected model without a saved example; see the scientific report for actual alternative-model and perturbation failures.'} The mapping is provisional, healthy controls are absent, and one internal test set cannot establish external clinical performance.

## What we'd do next

Independently verify original feature-order annotation and evaluate a patient-aware external cohort with compatible measurements. Study calibration and selected-feature stability using training-only resampling.

## Sources

[UCI dataset, DOI 10.24432/C5R88H](https://doi.org/10.24432/C5R88H); [provisional mapping](https://github.com/Roman-Young/tcga-pancan-tumor-classifier/blob/main/gene_mapping.csv); NCBI Gene/NHGRI biological resources; NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, Plotly and Streamlit. Codex/ChatGPT assisted with code and documentation; actual contributions are recorded separately.

**This project is a prototype and is not medical advice.**
"""
    paths = []
    for filename, content in (("SCIENTIFIC_REPORT.md", scientific), ("HACKATHON_WRITEUP.md", writeup)):
        path = report_dir / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        start, end = "<!-- MEASURED_RESULTS_START -->", "<!-- MEASURED_RESULTS_END -->"
        if start in content and end in content:
            block = f"{start}\n\n{measured_sentence} {baseline_sentence} {manual_sentence}\n\n{performance_table(test)}\n\n{end}"
            content = content[:content.index(start)] + block + content[content.index(end) + len(end):]
            readme.write_text(content, encoding="utf-8")
            paths.append(readme)
    return paths


if __name__ == "__main__":
    for written in generate_reports():
        print(written.relative_to(ROOT))
