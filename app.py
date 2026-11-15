"""GenePrint: an interactive research interface for saved tumor-expression classifiers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CLASS_NAMES, DATA_DIR, MODEL_DIR, REPORT_DIR, RESULT_DIR, SEED
from src.inference import load_bundle, load_registry, predict_values
from src.manual_inputs import make_csv_template, parse_manual_csv, validate_manual_values


NAVY = "#21496D"
GOLD = "#D4AF37"
COLORS = {"BRCA": "#397EAD", "COAD": "#BB8230", "KIRC": "#328570", "LUAD": "#9B6292", "PRAD": "#CF705B"}
PAGES = ["Home", "Analyze a Tumor Sample", "Try Your Own Gene Values", "Explore the Genes", "Our Findings", "Detailed Model Comparison", "How Reliable Is the Model?", "Research Methods", "Meet the Team"]
# Preserve the original navigation values for existing Streamlit UI tests,
# while displaying accessible names to visitors.
LEGACY_TO_PAGE = {
    "Home": "Home",
    "Sample Explorer": "Analyze a Tumor Sample",
    "Enter Gene Values": "Try Your Own Gene Values",
    "Model Comparison": "Detailed Model Comparison",
    "Gene Explorer": "Explore the Genes",
    "Robustness Lab": "How Reliable Is the Model?",
    "Research Results": "Our Findings",
    "About & Team": "Meet the Team",
    "Research Methods": "Research Methods",
}
NAV_CHOICES = list(LEGACY_TO_PAGE)
PAGE_TO_LEGACY = {display: old for old, display in LEGACY_TO_PAGE.items()}

MODEL_LABELS = {
    "majority": "Baseline — always predicts the largest category",
    "full": "All features — 20,531 measurements",
    "compact_10": "10-feature model — quick experiment",
    "compact_25": "25-gene model",
    "compact_50": "Recommended model — 50 measurements",
    "compact_100": "100-gene model",
    "compact_500": "500-gene model",
}

def sample_options(samples: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Give public examples stable, nonidentifying display names."""
    ids = list(samples.index)
    return ids, {sid: f"Research Sample {index + 1:02d}" for index, sid in enumerate(ids)}


def friendly_class(code: str) -> str:
    return CLASS_NAMES.get(code, code)

DISCLAIMER = "This project is a prototype and is not medical advice."
SCORE_NOTE = "These scores compare five tumor categories in a public research dataset—not whether someone has cancer."


def read_table(path: Path) -> pd.DataFrame:
    """Return an optional result table without hiding malformed existing files."""
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def cached_demo(path: str, modified: float) -> pd.DataFrame:
    """Cache public demonstration rows; modification time invalidates the cache."""
    return pd.read_csv(path, index_col="sample_id")


@st.cache_resource(show_spinner=False)
def cached_bundle(model_id: str, registry_modified: float) -> dict:
    """Load the trained pipeline once instead of retraining during page refreshes."""
    return load_bundle(model_id)


def bundle_for(model_id: str) -> dict:
    """Resolve a registered model and invalidate the cache after retraining."""
    return cached_bundle(model_id, (MODEL_DIR / "registry.json").stat().st_mtime)


def demo_samples() -> pd.DataFrame:
    """Load only the small public held-out sample bundle used by the application."""
    path = RESULT_DIR / "demo_samples.csv.gz"
    if not path.is_file():
        st.info("Public demonstration samples are being prepared. Complete the training command in README.md.")
        st.stop()
    return cached_demo(str(path), path.stat().st_mtime)


def chart_style(figure, height: int = 370):
    """Use accessible text and restrained gold-accented charts on the dark site."""
    figure.update_layout(
        height=height,
        font=dict(family="Arial, sans-serif", color="#EDF4FF"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=20, t=60, b=50), legend_title_text="",
    )
    figure.update_xaxes(gridcolor="#344B69", zerolinecolor="#45617F")
    figure.update_yaxes(gridcolor="#344B69", zerolinecolor="#45617F")
    return figure


def go_to(page: str):
    """Navigation callback runs before the sidebar widget is instantiated again."""
    st.session_state["page"] = PAGE_TO_LEGACY.get(page, page)


def section_intro(eyebrow: str, title: str, description: str):
    """Render a concise page introduction using trusted static application text."""
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)


def model_picker(registry: dict, key: str, compact_only: bool = False) -> str:
    """Preserve registered IDs internally and explain choices in ordinary language."""
    models = registry["models"]
    if compact_only:
        models = [m for m in models if m["model_id"] in ("compact_10", "compact_25")]
    ids = [m["model_id"] for m in models]
    labels = {m["model_id"]: MODEL_LABELS.get(m["model_id"], m.get("display_name", m["model_id"])) for m in models}
    if not ids:
        st.warning("The required trained model is not available.")
        st.stop()
    preferred = "compact_10" if compact_only else registry.get("selected_model", ids[0])
    index = ids.index(preferred) if preferred in ids else 0
    return st.selectbox("Choose an analysis method", ids, index=index, format_func=lambda value: labels[value], key=key)


def show_prediction(result: dict, manifest: dict, true_label: str | None = None):
    """Present saved-model predictions as research classification, not prognosis."""
    st.subheader("Your research result")
    columns = st.columns(3)
    columns[0].metric("Highest-ranked tumor category", friendly_class(result["predicted_label"]))
    columns[1].metric("Highest model score", f'{result["max_score"]:.1%}')
    columns[2].metric("Measurements analyzed", f'{len(manifest["input_features"]):,}')
    st.caption("The score describes this model's ranking among five tumor categories; it is not a personal cancer-risk estimate or a clinically validated confidence level.")
    if result.get("review_rule_unavailable"):
        st.warning("This baseline did not establish a qualifying uncertainty-review rule. It is included for comparison, not as a recommended classifier.")
    elif result["review_recommended"]:
        st.warning("The model flagged this research prediction for additional review. This is not clinical advice.")
    else:
        st.info("This prediction passes its research review rule. That does not guarantee correctness or clinical reliability.")
    scores = pd.DataFrame({"Tumor category": [friendly_class(k) for k in result["probabilities"]],
                           "Model score": list(result["probabilities"].values())})
    fig = px.bar(scores, x="Model score", y="Tumor category", orientation="h",
                 text=scores["Model score"].map(lambda x: f"{x:.1%}"),
                 color_discrete_sequence=[GOLD])
    fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(showlegend=False, title="How the five categories compare")
    st.plotly_chart(chart_style(fig, 350), use_container_width=True)
    if true_label is not None:
        correct = result["predicted_label"] == true_label
        st.write(f"**Known dataset label:** {friendly_class(true_label)}. "
                 f"The model {'matched' if correct else 'did not match'} this recorded label.")
    
    if result["out_of_range"]:
        st.warning(
            f"{len(result['out_of_range'])} gene activity value(s) "
            "fall outside the observed training range. "
            "Treat the output with extra caution."
        )

        with st.expander("Show out-of-range feature IDs"):
            st.write(", ".join(result["out_of_range"]))
    if result["missing_count"]:
        st.caption(f"{result['missing_count']} missing values were filled using the saved training-fitted imputer.")
    with st.expander("What do the scores and review flag mean?"):
        st.write("The largest model score determines the predicted category. Scores sum to one across the five categories, but have not been validated as clinical probabilities.")
        st.write(manifest.get("threshold_note", "The review rule was selected from validation results."))
        if result.get("review_rule_unavailable"):
            st.write("No qualifying review threshold was available for this baseline model.")
        else:
            st.write(f"Review threshold: {manifest.get('review_threshold', 1.0):.3f}. "
                     f"Difference between the top two scores: {result['margin']:.3f}.")
        st.write("Unusual combinations of otherwise in-range measurements may still be biologically unrealistic.")
    st.caption(SCORE_NOTE)


def home(registry: dict | None):
    """Public-first entry point with scientific scope and clear next steps."""
    st.markdown("""<div class="hero"><div class="eyebrow">GENEPRINT · CANCER GENE RESEARCH</div>
        <h1>Fewer measurements.<br><span>More accessible cancer research.</span></h1>
        <p>Could a small signature of gene activity distinguish tumor types almost as effectively as thousands of measurements? Explore the evidence—and test the models yourself.</p>
        <div class="hero-tags"><span>801 public tumor samples</span><span>Five tumor categories</span><span>Real held-out results</span></div>
        </div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.button("Analyze a Sample", on_click=go_to, args=("Analyze a Tumor Sample",), type="primary", use_container_width=True)
    c2.button("Try Gene Values", on_click=go_to, args=("Try Your Own Gene Values",), use_container_width=True)
    c3.button("Explore Our Findings", on_click=go_to, args=("Our Findings",), use_container_width=True)
    st.write("")
    st.subheader("Why we built GenePrint")
    st.markdown("""<div class="info-grid">
      <div class="info-card"><div class="card-kicker">01 · THE WORLDWIDE CHALLENGE</div><h3>Where someone lives can shape their options.</h3><p>Advanced cancer services and molecular expertise are not equally available across countries and communities. At the same time, expression-based research can involve thousands of measurements. Our question: how much information is actually needed for this five-type classification problem?</p></div>
      <div class="info-card"><div class="card-kicker">02 · WHAT WE BUILT</div><h3>One dataset. Seven ways to analyze it.</h3><p>GenePrint compares a simple baseline, a full 20,531-feature classifier, and five compact alternatives using 10, 25, 50, 100 or 500 selected measurements. Visitors can test real public examples and inspect the results.</p></div>
      <div class="info-card"><div class="card-kicker">03 · WHY IT COULD MATTER</div><h3>Make every measurement count.</h3><p>If reliable, small molecular signatures might inform future targeted assays and reduce unnecessary measurement and analysis burden. This is a research direction—not a measured reduction in laboratory cost.</p></div>
      <div class="info-card"><div class="card-kicker">04 · WHO COULD BENEFIT, EVENTUALLY?</div><h3>More inclusive molecular research.</h3><p>Researchers in regional labs, resource-constrained institutions, and training programs could explore compact approaches if future studies validate them across countries, populations, instruments and laboratories.</p></div>
    </div>""", unsafe_allow_html=True)
    st.write("")
    st.subheader("The global problem, in numbers")
    ca, cb = st.columns(2)
    ca.metric("New cancer cases worldwide (2024)", "20.6 million")
    cb.metric("Cancer deaths worldwide (2024)", "9.8 million")
    st.caption("Estimates from the International Agency for Research on Cancer, published July 2026. [View the global study](https://www.iarc.who.int/news-events/global-cancer-statistics-2024-globocan-estimates-of-incidence-and-mortality-worldwide-for-34-cancers-in-186-countries).")
    st.subheader("Where could smaller signatures make a difference?")
    audience = st.selectbox("Explore a possible future use", [
        "Regional reference laboratories", "Under-resourced research institutions", "Students and training programs"
    ], key="impact_audience")
    scenarios = {
        "Regional reference laboratories": "A future validated targeted panel might let centralized laboratories examine a limited set of measurements for a defined classification task. That would still require tissue samples, laboratory instrumentation, quality assurance and independent clinical evaluation.",
        "Under-resourced research institutions": "Compact candidate feature lists can help researchers prioritize experiments, compare computational methods, and study transferability without first developing a much larger feature-based model. Actual laboratory affordability remains to be tested.",
        "Students and training programs": "Public-data demonstrations make supervised learning, gene expression, class imbalance and uncertainty accessible without collecting patient information or requiring a wet lab."
    }
    st.info(scenarios[audience])
    with st.expander("What would real global access require?"):
        st.markdown("**Representative evidence:** Test samples from different regions, population backgrounds, hospitals and sequencing instruments.\n\n**Verified biological measurements:** Confirm each selected feature's identity and demonstrate a reproducible targeted measurement method.\n\n**Affordability studies:** Measure total costs for sample collection, reagents, personnel, equipment, quality assurance and interpretation—not just the feature count.\n\n**Local partnership:** Evaluate workflows with the laboratories and communities they are intended to serve.")
    st.subheader("Explore the feature-efficiency experiment")
    st.write("Choose how many original expression measurements a classifier uses and compare its recorded results. These scores come from saved experiments, not live estimates.")
    counts = [10, 25, 50, 100, 500, 20531]
    selected_count = st.select_slider("Number of gene-expression features", options=counts, value=50,
                                      format_func=lambda count: f"{count:,} features", key="home_feature_count")
    model_lookup = {10: "compact_10", 25: "compact_25", 50: "compact_50", 100: "compact_100", 500: "compact_500", 20531: "full"}
    current_name = model_lookup[selected_count]
    left, middle, right = st.columns(3)
    left.metric("Selected measurements", f"{selected_count:,}")
    middle.metric("Fewer inputs than full model", f"{(1 - selected_count / 20531):.2%}")
    summary_here = read_table(RESULT_DIR / "metrics/summary.csv")
    relevant = summary_here[(summary_here["model_id"] == current_name) & (summary_here["split"] == "test")] if not summary_here.empty else pd.DataFrame()
    if not relevant.empty:
        row = relevant.iloc[0]
        right.metric("Held-out test accuracy", f"{float(row['accuracy']):.2%}")
        st.caption(f"Model: {MODEL_LABELS.get(current_name, current_name)} · Macro-F1: {float(row['macro_f1']):.4f} · {int(row['n_samples'])} held-out samples. Selection of the recommended model used validation data, not test data.")
    else:
        right.metric("Held-out test accuracy", "Not available")
    st.caption("Changing the feature count here selects an existing trained model; it does not retrain one. Fewer inputs do not automatically mean proportionally lower testing costs.")
    st.write("")
    st.subheader("What is gene expression?")
    st.write("DNA carries a cell's instructions. When a gene is active, the cell makes RNA from it. Measuring RNA helps researchers estimate the activity of thousands of genes. Tumors originating in different tissues can have distinct patterns of gene activity.")
    meta_path = DATA_DIR / "manifest.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        a, b, c = st.columns(3)
        a.metric("Public tumor samples", f'{meta["samples"]:,}')
        b.metric("Original expression measurements", f'{meta["features"]:,}')
        c.metric("Tumor categories", len(meta["class_counts"]))
    else:
        a, b, c = st.columns(3)
        a.metric("Public tumor samples", "801")
        b.metric("Original expression measurements", "20,531")
        c.metric("Tumor categories", "5")
    if registry:
        summary = read_table(RESULT_DIR / "metrics/summary.csv")
        if not summary.empty:
            selected = summary[summary["model_id"] == registry.get("selected_model")]
            if not selected.empty:
                split = "test" if "test" in selected["split"].values else "validation"
                row = selected[selected["split"] == split].iloc[0]
                st.caption(f"Research snapshot · Selected model: {registry['selected_model']} · {split} accuracy: {row['accuracy']:.1%} · macro-F1: {row['macro_f1']:.3f}. Detailed comparisons appear in Our Findings.")
    st.caption("Scope: GenePrint compares five kinds of existing research tumor samples. It cannot test for cancer, predict prognosis or guide treatment.")


def sample_explorer(registry: dict):
    """Public sample flow with human-readable labels and preserved dataset IDs."""
    section_intro("EXPLORE · STEP BY STEP", "Analyze a Tumor Sample",
                  "Choose a public research tumor sample, select a trained model, and see which of five tumor categories its gene-activity pattern most closely resembles.")
    st.info("You are analyzing an existing research tumor sample—not submitting a medical test or checking whether someone has cancer.")
    samples = demo_samples()
    ids, labels = sample_options(samples)
    st.subheader("1 · Choose a research example")
    sample_id = st.selectbox("Choose a sample to explore", ids, format_func=lambda x: labels[x], key="sample_id")
    with st.expander("Show original dataset identifier"):
        st.code(str(sample_id))
        st.caption("These numbered examples are display labels. Every example uses its original saved measurements.")
    st.subheader("2 · Choose an analysis method")
    model_id = model_picker(registry, "sample_model")
    if model_id == "majority":
        st.caption("The baseline always predicts the most common tumor class. It serves as a comparison, not a recommended analysis tool.")
    else:
        st.caption("The compact models use selected gene measurements; the full model uses the original feature set. All choices load already-trained classifiers.")
    show_label = st.checkbox("Compare with the known research label", value=True)
    st.subheader("3 · See the model's result")
    st.write("You'll receive the closest matching tumor type, a score for each of the five categories, and notes about uncertainty.")
    if st.button("Analyze Sample", type="primary", use_container_width=True):
        bundle = bundle_for(model_id)
        sample = samples.loc[sample_id]
        values = sample[bundle["manifest"]["input_features"]].to_numpy(dtype=float)
        show_prediction(predict_values(bundle, values), bundle["manifest"],
                        str(sample["cancer_type"]) if show_label else None)


def set_manual_fields(model_id: str, feature_ids: list[str], values: list):
    """Reset/clear callback writes input state before the form is instantiated."""
    for feature, value in zip(feature_ids, values):
        st.session_state[f"manual_{model_id}_{feature}"] = "" if value is None else str(value)


def manual_input(registry: dict):
    """Research-only manual experiment that preserves exact feature order and validation."""
    section_intro("INTERACTIVE EXPERIMENT", "Try Your Own Gene Values",
                  "Start with a real public tumor example, change individual gene-activity measurements, and watch how a trained model's classification responds.")
    st.info("These numbers are measurements on the research dataset's scale—not percentages, personal health results, or inputs from a home test. Please don't enter personal medical information.")
    model_id = model_picker(registry, "manual_model", compact_only=True)
    bundle = bundle_for(model_id)
    manifest, features = bundle["manifest"], bundle["features"]
    names = manifest["input_features"]
    st.write(f"This version uses **{len(names)} gene-activity measurements** in a fixed, verified feature order. The model was trained before you opened this page.")
    samples = demo_samples()
    ids, labels = sample_options(samples)
    st.subheader("1 · Start with a research example")
    example_id = st.selectbox("Choose a sample to start with", ids, format_func=lambda x: labels[x], key="manual_example")
    example = samples.loc[example_id, names].to_numpy(dtype=float)
    a, b, c = st.columns(3)
    a.button("Load / Restore Example", key="reset_manual", on_click=set_manual_fields,
             args=(model_id, names, example.tolist()), use_container_width=True)
    b.button("Clear Inputs", key="clear_manual", on_click=set_manual_fields,
             args=(model_id, names, [None] * len(names)), use_container_width=True)
    c.download_button("Download CSV Template", make_csv_template(names),
                      file_name=f"{model_id}_input_template.csv", mime="text/csv", use_container_width=True)
    with st.expander("What do these gene-activity numbers mean?"):
        st.write("A gene-expression value estimates how active a measured gene was in the original RNA-sequencing dataset. Larger values indicate higher expression on that dataset's supplied scale. These are not percentages or gene mutation counts.")
        st.write("Starting from an example lets you explore one change at a time. Invented combinations may not represent biologically possible samples. Only matching research-dataset measurements should be compared.")
    st.subheader("2 · Review or edit the measurements")
    st.caption("The candidate biological gene names are provisional. Original feature identifiers stay available so the values enter the right model inputs.")
    annotations = read_table(RESULT_DIR / "experiments/biological_annotations.csv")
    with st.form(f"manual_form_{model_id}"):
        columns = st.columns(2)
        entered = {}
        for index, row in features.iterrows():
            feature = row["uci_feature"]
            symbol = str(row["gene_symbol"]).strip() or f"Gene feature {index + 1}"
            with columns[index % 2]:
                entered[feature] = st.text_input(
                    f"{symbol} — gene activity", key=f"manual_{model_id}_{feature}",
                    placeholder="Enter an expression measurement",
                    help=f"Dataset feature ID: {feature}. Candidate annotation status: {row['mapping_status']}.")
                st.caption(f"Feature {feature} · Observed training range: {float(row['train_min']):.3g} to {float(row['train_max']):.3g} · Median: {float(row['train_median']):.3g}")
                if not annotations.empty:
                    matched = annotations[annotations["uci_feature"] == feature]
                    if not matched.empty and pd.notna(matched.iloc[0].get("biological_function")):
                        biology = matched.iloc[0]
                        st.caption(str(biology["biological_function"]) + f" [Source]({biology['source_url']})")
        analyze = st.form_submit_button("Analyze Gene Values", type="primary", use_container_width=True)
    st.subheader("3 · Understand the result")
    st.caption("The model estimates which of five tumor categories the entered numerical pattern most resembles; it cannot diagnose or predict outcomes for an individual.")
    if analyze:
        try:
            values = validate_manual_values(entered, names)
            if not np.array_equal(values[0], example):
                st.info("You modified the example. Edited values may not represent a realistic biological sample.")
            show_prediction(predict_values(bundle, values), manifest)
        except (ValueError, TypeError) as error:
            st.error(f"Please check your measurements: {error}")
    with st.expander("Advanced: analyze a one-row CSV"):
        st.write("Provide exactly the original feature IDs as CSV headers. The app aligns matching columns by ID; it does not guess missing values.")
        upload = st.file_uploader("Research expression CSV", type=["csv"], key=f"upload_{model_id}")
        if st.button("Analyze Uploaded CSV", key=f"analyze_csv_{model_id}"):
            if upload is None:
                st.error("Choose a CSV file first.")
            else:
                try:
                    values, reordered = parse_manual_csv(upload.getvalue(), names)
                    if reordered:
                        st.info("Columns were aligned to the saved model's expected gene IDs.")
                    show_prediction(predict_values(bundle, values), manifest)
                except (ValueError, TypeError) as error:
                    st.error(f"Please check your CSV: {error}")
    st.caption(SCORE_NOTE)


def model_comparison(registry: dict):
    """Compare measured validation/test performance without combining the partitions."""
    section_intro("BEHIND THE RESEARCH", "Model Performance", "How do simpler models compare with the full model? These graphs use real saved evaluation results, not predictions invented for this page.")
    st.caption("Accuracy is the percentage of correct predictions. Macro-F1 gives every tumor category equal weight, including those with fewer examples.")
    summary = read_table(RESULT_DIR / "metrics/summary.csv")
    if summary.empty:
        st.info("No completed model evaluations are available yet.")
        return
    split = st.radio("Evaluation partition", sorted(summary["split"].unique()), horizontal=True, key="comparison_split")
    st.caption("Validation guides selection. Final test results are measured only after model choices and the evaluation protocol are locked.")
    current = summary[summary["split"] == split].copy()
    metric = st.selectbox("Metric", ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1"], key="comparison_metric")
    figure = px.bar(current, x="model_id", y=metric, color="model_id", text=current[metric].map(lambda x: f"{x:.3f}"),
                    color_discrete_sequence=["#D4AF37", "#8FC1E3", "#CDAA70", "#78A5BB", "#E8C66A", "#6F8EAF", "#4F7198"],
                    labels={"model_id": "Trained model", metric: metric.replace("_", " ").title()})
    figure.update_yaxes(range=[0, 1.08])
    figure.update_layout(title=f"{metric.replace('_', ' ').title()} · {split} · n={int(current.iloc[0]['n_samples'])}", showlegend=False)
    st.plotly_chart(chart_style(figure), use_container_width=True)
    st.dataframe(current, hide_index=True, use_container_width=True)
    st.download_button("Download measured metrics", summary.to_csv(index=False), "geneprint_metrics.csv", "text/csv")
    compare_ids = current["model_id"].tolist()
    preferred_compare = registry.get("selected_model")
    selected = st.selectbox("Inspect per-class results", compare_ids,
                            index=compare_ids.index(preferred_compare) if preferred_compare in compare_ids else 0,
                            key="comparison_model", format_func=lambda mid: MODEL_LABELS.get(mid, mid))
    per_class = read_table(RESULT_DIR / "metrics/per_class.csv")
    if not per_class.empty:
        details = per_class[(per_class["model_id"] == selected) & (per_class["split"] == split)]
        if not details.empty:
            a, b = st.columns(2)
            with a:
                long = details.melt(id_vars="class", value_vars=["precision", "recall", "f1"], var_name="Metric", value_name="Score")
                figure = px.bar(long, x="class", y="Score", color="Metric", barmode="group", labels={"class": "Tumor category"}, title="Per-class performance")
                figure.update_yaxes(range=[0, 1.08])
                st.plotly_chart(chart_style(figure, 330), use_container_width=True)
            with b:
                long = details.melt(id_vars="class", value_vars=["false_positives", "false_negatives"], var_name="Error type", value_name="Samples")
                figure = px.bar(long, x="class", y="Samples", color="Error type", barmode="group", labels={"class": "Tumor category"}, title="One-versus-rest error counts")
                st.plotly_chart(chart_style(figure, 330), use_container_width=True)
            st.dataframe(details, hide_index=True, use_container_width=True)
    path = RESULT_DIR / "metrics/confusion_matrices.json"
    if path.is_file():
        matrices = json.loads(path.read_text())
        matrix = matrices.get(selected, {}).get(split)
        if matrix:
            figure = px.imshow(np.asarray(matrix["matrix"]), x=matrix["labels"], y=matrix["labels"], text_auto=True,
                               labels={"x": "Predicted tumor category", "y": "True tumor category", "color": "Samples"},
                               color_continuous_scale=["#F7F7F2", "#B7C5D9", NAVY], title=f"Confusion matrix · {selected} · {split}")
            st.plotly_chart(chart_style(figure, 420), use_container_width=True)
    st.caption("False positive for a category: another tumor category predicted as that category. False negative: a sample of that category predicted as another. These are tumor-type errors, not healthy-versus-cancer errors.")


def gene_explorer(registry: dict):
    """Explore selected original features, provisional annotations and training distributions."""
    section_intro("UNDERSTAND THE BIOLOGY", "Explore the Genes", "Discover which gene-expression features the models use, what their candidate biological names mean, and how their measurements differ across tumor types.")
    st.info("A gene can help distinguish tumor samples without causing cancer. Tissue identity alone can influence expression patterns.")
    model_id = model_picker(registry, "gene_model")
    features = bundle_for(model_id)["features"].copy()
    query = st.text_input("Search by original gene ID, symbol or Entrez ID", key="gene_search").strip()
    if query:
        mask = features[["uci_feature", "gene_symbol", "entrez_id"]].astype(str).apply(lambda c: c.str.contains(query, case=False, regex=False)).any(axis=1)
        display = features[mask]
    else:
        display = features
    st.caption(f"{len(display):,} matching features. Third-party mapping is structurally checked; complete independent TCGA index verification remains pending.")
    with st.expander("See the complete technical feature table"):
        st.dataframe(display, hide_index=True, use_container_width=True)
    st.download_button("Download selected feature crosswalk", features.to_csv(index=False), f"{model_id}_features.csv", "text/csv")
    if display.empty:
        return
    feature_labels = {r["uci_feature"]: (f"{r['gene_symbol']} · {r['uci_feature']}" if str(r["gene_symbol"]).strip() else r["uci_feature"])
                      for _, r in display.iterrows()}
    feature = st.selectbox("Choose a gene feature to explore", display["uci_feature"].tolist(), key="gene_feature", format_func=lambda x: feature_labels[x])
    row = features[features["uci_feature"] == feature].iloc[0]
    st.write(f"**{row['gene_symbol'] or 'Unresolved symbol'}** · {feature} · Entrez {row['entrez_id']}")
    st.markdown(f"[NCBI Gene reference](https://www.ncbi.nlm.nih.gov/gene/{row['entrez_id']})")
    annotations = read_table(RESULT_DIR / "experiments/biological_annotations.csv")
    annotation = annotations[annotations["uci_feature"] == feature] if not annotations.empty else pd.DataFrame()
    if not annotation.empty:
        record = annotation.iloc[0]
        st.write(str(record["biological_function"]))
        st.write(str(record["interpretation"]))
        st.markdown(f"[Biological source]({record['source_url']})")
    else:
        st.caption("A source-backed biological description has not been curated for this feature. The identifier link does not independently verify its position in the UCI matrix.")
    expression = read_table(RESULT_DIR / "experiments/gene_expression.csv")
    if not expression.empty:
        selected = expression[expression["uci_feature"] == feature]
        if not selected.empty:
            figure = px.box(selected, x="cancer_type", y="value", color="cancer_type", color_discrete_map=COLORS,
                            labels={"cancer_type": "Tumor category", "value": "Supplied expression scale"},
                            title=f"{feature} expression · training samples only · n={selected['sample_id'].nunique()}")
            figure.update_layout(showlegend=False)
            st.plotly_chart(chart_style(figure), use_container_width=True)
        else:
            st.caption("A distribution export is available only for the curated compact signatures.")
    with st.expander("What does a selected gene actually tell us?"):
        st.write("A selected feature helps the model statistically distinguish tissue-derived tumor categories. Selection does not mean that gene causes the tumor or that a new biomarker was discovered. Candidate biological names need independent mapping verification.")


def robustness_lab(registry: dict):
    """Keep live single-sample perturbations distinct from completed aggregate experiments."""
    section_intro("HOW STABLE ARE THE PREDICTIONS?", "How Reliable Is the Model?", "Explore what happens when we deliberately change or hide some gene-activity measurements in a simulated research experiment.")
    st.write("A robust model should ideally keep making accurate classifications even when measurements are slightly imperfect. These are synthetic stress tests, not direct tests of a medical laboratory.")
    st.caption("The simulated changes demonstrate mathematical sensitivity, not real-world laboratory validation.")
    aggregate = read_table(RESULT_DIR / "experiments/robustness.csv")
    if not aggregate.empty:
        st.subheader("Completed aggregate experiments")
        experiment = st.selectbox("Experiment", aggregate["experiment"].unique().tolist(), key="robust_experiment")
        metric = st.selectbox("Aggregate performance metric", ["macro_f1", "accuracy", "balanced_accuracy"], key="robust_metric")
        subset = aggregate[aggregate["experiment"] == experiment]
        grouped = subset.groupby(["model_id", "level"])[metric].agg(["mean", "std"]).reset_index()
        grouped["std"] = grouped["std"].fillna(0)
        figure = px.line(grouped, x="level", y="mean", color="model_id", markers=True, error_y="std",
                         labels={"level": "Perturbation level", "mean": metric.replace("_", " "), "model_id": "Model"},
                         title=f"{experiment} · completed held-out evaluations")
        figure.update_yaxes(range=[0, 1.05])
        st.plotly_chart(chart_style(figure), use_container_width=True)
        st.caption("Points show the mean over recorded seeds; error bars show their standard deviation, not a confidence interval. Only measured settings are shown.")
        st.download_button("Download robustness measurements", aggregate.to_csv(index=False), "robustness.csv", "text/csv")
    st.subheader("Live perturbation · one public sample")
    model_id = model_picker(registry, "robust_model")
    bundle = bundle_for(model_id)
    samples = demo_samples()
    robust_ids, robust_labels = sample_options(samples)
    sample_id = st.selectbox("Choose a research example", robust_ids, key="robust_sample", format_func=lambda sid: robust_labels[sid])
    noise = st.slider("Amount of simulated measurement error", 0.0, 3.0, 0.0, 0.1)
    missing = st.select_slider("Percentage of measurements hidden", options=[0.0, 0.05, 0.10, 0.20], value=0.0)
    with st.expander("Advanced experiment settings"):
        seed = st.number_input("Repeatable experiment seed", value=SEED, min_value=0, max_value=2**31 - 1, step=1)
        st.caption("Gaussian noise uses training-derived per-gene scale. Missing measurements use the saved training-fitted imputer. Simulated negative values are retained and flagged; they are not clipped.")
    if st.button("Run Perturbation", type="primary"):
        from src.robustness import add_noise, mask_missing
        values = samples.loc[sample_id, bundle["manifest"]["input_features"]].to_numpy(dtype=float).reshape(1, -1)
        perturbed = add_noise(values, bundle["features"]["train_std"].to_numpy(dtype=float), noise, seed=int(seed))
        perturbed = mask_missing(perturbed, missing, seed=int(seed) + 1)
        original = predict_values(bundle, values)
        result = predict_values(bundle, perturbed, allow_missing=True)
        st.write(f"Unperturbed prediction: **{original['predicted_label']}**, score {original['max_score']:.1%}. "
                 f"Live perturbation changed {int(np.isnan(perturbed).sum())} measurements to missing.")
        show_prediction(result, bundle["manifest"], str(samples.loc[sample_id, "cancer_type"]))
        st.caption("This one-sample interaction is not an estimate of aggregate accuracy and is not added to the saved experiment results.")


def research_results(registry: dict):
    """Show the measured study outputs and genuine error records with downloadable reports."""
    section_intro("EVIDENCE & FINDINGS", "Our Findings", "We compared models using different numbers of gene-activity measurements to understand how model complexity relates to classification performance.")
    st.button("View detailed model comparisons, class errors and confusion matrices", on_click=go_to, args=("Detailed Model Comparison",), use_container_width=True)
    st.info("Start here for the scientific evidence. All metrics below come from saved experiments, and validation and final test results are identified separately.")
    st.write("A smaller model uses fewer measured features. This may be useful for future research, but does not by itself establish a cheaper or clinically reliable test.")
    st.write("**Research question:** Can compact signatures preserve tumor-type performance while remaining reliable under simulated measurement noise?")
    st.write("**Hypotheses:** compact signatures may retain predictive performance; feature count may affect perturbation sensitivity; some selected features may relate to established tissue biology. These are tested propositions, not assumed conclusions.")
    st.write("**Design:** stratified training, validation and held-out test partitions; training-fitted imputation, supervised feature selection and scaling; majority baseline and regularized logistic regression; a locked final evaluation protocol.")
    summary = read_table(RESULT_DIR / "metrics/summary.csv")
    if not summary.empty:
        st.subheader("Compare two approaches")
        available = [item for item in ["compact_10", "compact_25", "compact_50", "compact_100", "compact_500", "full", "majority"] if item in summary["model_id"].values]
        first, second = st.columns(2)
        first_id = first.selectbox("First analysis approach", available, index=available.index("compact_50") if "compact_50" in available else 0,
                                   format_func=lambda mid: MODEL_LABELS.get(mid, mid), key="compare_one")
        second_id = second.selectbox("Compare against", available, index=available.index("full") if "full" in available else min(1, len(available)-1),
                                    format_func=lambda mid: MODEL_LABELS.get(mid, mid), key="compare_two")
        a, b = st.columns(2)
        for box, model_id in [(a, first_id), (b, second_id)]:
            measured = summary[(summary["model_id"] == model_id) & (summary["split"] == "test")]
            if not measured.empty:
                r = measured.iloc[0]
                box.metric(f"{int(r['n_features']):,} features", f"{r['accuracy']:.2%} test accuracy")
                box.caption(f"Macro-F1: {r['macro_f1']:.4f} · {int(r['n_samples'])} test examples")
    if not summary.empty:
        with st.expander("View the full numerical results table"):
            st.dataframe(summary, hide_index=True, use_container_width=True)
        metric = st.selectbox("Feature-count graph", ["accuracy", "macro_f1", "balanced_accuracy"], key="research_metric")
        split = st.radio("Feature-count evaluation partition", sorted(summary["split"].unique()), horizontal=True, key="research_split")
        curve = summary[(summary["split"] == split) & (~summary["model_id"].str.contains("majority|dummy|baseline", case=False, regex=True))].sort_values("n_features")
        figure = px.line(curve, x="n_features", y=metric, markers=True, log_x=True,
                         labels={"n_features": "Measured gene-feature count (log scale)", metric: metric.replace("_", " ")},
                         title=f"Feature count and performance · {split}")
        figure.update_yaxes(range=[0, 1.05])
        figure.update_xaxes(tickmode="array", tickvals=[10, 25, 50, 100, 500, 20531],
                            ticktext=["10", "25", "50", "100", "500", "20,531"])
        st.plotly_chart(chart_style(figure), use_container_width=True)
    st.subheader("Genuine model errors")
    prediction_frames = []
    for split in ("validation", "test"):
        frame = read_table(RESULT_DIR / f"predictions/{split}_predictions.csv")
        if not frame.empty:
            frame["split"] = split
            prediction_frames.append(frame)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        error_ids = predictions["model_id"].unique().tolist()
        preferred_error = registry.get("selected_model", "compact_50")
        selected = st.selectbox("Model for error inspection", error_ids,
                                index=error_ids.index(preferred_error) if preferred_error in error_ids else 0,
                                key="error_model", format_func=lambda mid: MODEL_LABELS.get(mid, mid))
        records = predictions[predictions["model_id"] == selected]
        errors = records[records["true_label"] != records["predicted_label"]]
        if errors.empty:
            st.info("This model has no errors in the saved clean evaluation rows. That does not establish perfect generalization. See the separately labeled robustness experiments for perturbation failures.")
        else:
            columns = [c for c in ["sample_id", "split", "true_label", "predicted_label", "max_score", "model_id"] if c in errors]
            st.dataframe(errors[columns], hide_index=True, use_container_width=True)
            st.caption("These are observed errors. Ambiguous expression patterns, finite training data and model limitations are possible explanations; they are not demonstrated causes for individual samples.")
    st.caption("Methods and biological sources are available on the Research Methods page. Research results describe the supplied tumor dataset and its held-out evaluation.")


def about():
    """Introduce the student research team without inventing credentials or roles."""
    section_intro("THE PEOPLE BEHIND THE PROJECT", "Meet the Team",
                  "Three aspiring student researchers exploring the intersection of biology, computation and healthcare.")
    st.markdown("""<div class="team-card"><div class="card-kicker">GENEPRINT · STUDENT RESEARCH</div>
      <h3>Three Student Researchers · One Shared Goal</h3>
      <p>GenePrint was developed by three aspiring student researchers united by curiosity about molecular biology, data science, and global health equity.</p>
      <p>Our team believes meaningful progress begins with curiosity, careful experimentation, and a commitment to making complex ideas accessible.</p>
      <p>Through GenePrint, we investigated whether a much smaller set of gene-activity measurements could retain useful tumor-type classification performance—and how stable these results remain when measurements change.</p>
      <p>Our long-term goal is to help build a future in which meaningful scientific tools and molecular research are more accessible across communities, institutions, and borders.</p></div>""", unsafe_allow_html=True)
    st.subheader("Our approach")
    st.write("We built an interactive demonstration using public RNA-sequencing research data. The models classify tumor samples among five known categories, compare different numbers of features, and examine the effects of simulated measurement changes.")
    st.subheader("Our vision")
    st.write("A future in which a limited, validated molecular signature could help research laboratories focus their assays, reduce unnecessary analysis, and broaden access to molecular science. Our present work is an early computational step toward that goal.")


def research_methods(registry: dict | None):
    """Readable summary of the study with optional scientific detail."""
    section_intro("HOW THE PROJECT WORKS", "Research Methods",
                  "From public tumor measurements to reproducible classification experiments.")
    steps = [
        ("1. Start with public research data", "We used the UCI Gene Expression Cancer RNA-Seq dataset: 801 previously collected tumor samples, 20,531 gene-expression features, and five known tumor categories."),
        ("2. Separate the samples", "We divided the data into training, validation and held-out testing portions. Training learns patterns, validation selects a model, and the final test checks those locked choices."),
        ("3. Learn from gene-activity patterns", "The main classifiers use regularized logistic regression. Training-fitted preprocessing and supervised feature selection are applied without using test data to choose the model."),
        ("4. Compare simpler models", "We compared a majority-class baseline and a full-feature classifier with compact models using 10, 25, 50, 100 and 500 selected measurements."),
        ("5. Test reliability", "The research examines errors by tumor category and simulates measurement noise or missing values to explore sensitivity to imperfect inputs."),
        ("6. Show limitations", "The model has not been validated with healthy controls, independent hospitals or clinical patient outcomes. Some biological gene annotations remain provisional."),
    ]
    for title, description in steps:
        st.subheader(title)
        st.write(description)
    with st.expander("See the five tumor categories"):
        for code, name in CLASS_NAMES.items():
            st.write(f"**{code}** — {name}")
    with st.expander("Research references"):
        st.subheader("Software and methods")
        st.markdown("[Scikit-learn: logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) · [Scikit-learn: feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) · [Scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) · [Streamlit documentation](https://docs.streamlit.io/) · [Plotly Python](https://plotly.com/python/)")
        st.subheader("Biology")
        st.markdown("[NHGRI: gene expression](https://www.genome.gov/genetics-glossary/Gene-Expression) · [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/) · [IARC 2024 global cancer estimates](https://www.iarc.who.int/news-events/global-cancer-statistics-2024-globocan-estimates-of-incidence-and-mortality-worldwide-for-34-cancers-in-186-countries)")
    st.caption("The source dataset contains previously collected tumor samples; this site is not a clinical diagnostic service.")


def main():
    """Public-first navigation with research detail available for judges."""
    st.set_page_config(page_title="GenePrint · Cancer Gene Research", page_icon="🧬", layout="wide")
    st.markdown("""<style>
      :root {color-scheme:dark;}
      .stApp {background:radial-gradient(circle at 85% 0%,#18365A 0%,#0B203A 40%,#07182D 100%);color:#F3F5FA;}
      .block-container {max-width:1220px;padding-top:2rem;padding-bottom:3rem;}
      h1,h2,h3 {letter-spacing:-.03em;color:#F8FAFF!important;}
      h1 {font-weight:800!important;}
      p,li,label,[data-testid="stMarkdownContainer"] {color:#DCE7F4;}
      [data-testid="stSidebar"] {background:#06152A;border-right:1px solid #314A66;}
      [data-testid="stSidebar"] * {color:#EDF3FD;}
      [data-testid="stSidebar"] button {text-align:left;justify-content:flex-start;border:0!important;background:transparent!important;color:#DDE8F5!important;padding:.4rem .7rem!important;}
      [data-testid="stSidebar"] button:hover {background:#1A3554!important;color:#F2CB67!important;}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#BFD0E4;}
      .eyebrow,.card-kicker {font-size:.74rem;letter-spacing:.13em;font-weight:800;color:#E8C66A;margin-bottom:.6rem;}
      .hero {background:linear-gradient(125deg,#0E2D50,#0A1A30);border:1px solid #42618A;border-top:4px solid #D4AF37; border-radius:20px;padding:3rem;box-shadow:0 18px 45px #0002;}
      .hero h1 {font-size:clamp(2.5rem,4.3vw,4rem);line-height:1.09;max-width:900px;margin:0 0 1.2rem;}
      .hero h1 span {color:#E8C66A;}
      .hero p {color:#DBE7F5;max-width:790px;font-size:1.12rem;line-height:1.7;}
      .hero-tags {display:flex;flex-wrap:wrap;gap:.6rem;margin-top:1.3rem;}
      .hero-tags span {border:1px solid #5C7291;border-radius:25px;padding:.35rem .75rem;font-size:.72rem;color:#E7EFFA;}
      .info-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin:1rem 0 2rem;}
      .info-card,.team-card {background:linear-gradient(145deg,#153455,#102841);border:1px solid #37536D;border-radius:15px;padding:1.5rem;}
      .info-card h3,.team-card h3 {font-size:1.18rem;color:#F2D27B!important;margin:.35rem 0 .65rem;}
      .info-card p,.team-card p {line-height:1.65;color:#DCE7F4;}
      .team-card {margin:.6rem 0 2rem;padding:2rem;}
      [data-testid="stMetric"] {background:#142C48;border:1px solid #35526D;border-radius:13px;padding:1rem;margin:.3rem 0;}
      [data-testid="stMetricValue"] {color:#F2CF75!important;font-size:1.55rem!important;}
      [data-testid="stMetricLabel"] {color:#D4E0EF!important;}
      [data-testid="stAlert"] {border-radius:12px;}
      div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {background:#D4AF37;color:#091B30;border:1px solid #E8C66A;font-weight:750;}
      div[data-testid="stButton"] button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {background:#E9C86D;color:#091B30;}
      [data-testid="stDataFrame"] {border:1px solid #385473;border-radius:10px;overflow:hidden;}
      .brand {font-size:1.08rem;font-weight:850;letter-spacing:.07em;color:#FFFFFF;}
      .brand-sub {font-size:.66rem;letter-spacing:.18em;font-weight:800;color:#D4AF37;margin:.3rem 0 1.5rem;}
      .nav-section {font-size:.68rem;letter-spacing:.15em;color:#D8B452!important;font-weight:800;margin-top:1.15rem;margin-bottom:.45rem;}
      @media(max-width:750px){.hero{padding:1.5rem}.info-grid{grid-template-columns:1fr}.block-container{padding-top:1rem}}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand">GENEPRINT</div><div class="brand-sub">CANCER GENE RESEARCH</div>', unsafe_allow_html=True)
        if "page" not in st.session_state or st.session_state["page"] not in NAV_CHOICES:
            st.session_state["page"] = "Home"
        st.markdown('<div class="nav-section">EXPLORE · BEHIND THE RESEARCH</div>', unsafe_allow_html=True)
        st.radio(
            "Research navigation",
            NAV_CHOICES,
            key="page",
            format_func=lambda value: LEGACY_TO_PAGE[value],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Five tumor categories · Public research data · Transparent experiments")
        st.caption("Explore the research · Compare real results")
    page = LEGACY_TO_PAGE[st.session_state["page"]]
    try:
        registry = load_registry()
    except FileNotFoundError:
        registry = None
    try:
        if page == "Home":
            home(registry)
        elif page == "Meet the Team":
            about()
        elif page == "Research Methods":
            research_methods(registry)
        elif registry is None:
            st.warning("Trained research models are not available. Check your local model files.")
        else:
            {
                "Analyze a Tumor Sample": sample_explorer,
                "Try Your Own Gene Values": manual_input,
                "Explore the Genes": gene_explorer,
                "Our Findings": research_results,
                "Detailed Model Comparison": model_comparison,
                "How Reliable Is the Model?": robustness_lab,
            }[page](registry)
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(f"We couldn't open the saved research information for this page: {error}")
        st.info("Check the model or results files. No prediction has been invented or substituted.")
    st.divider()
    st.caption(DISCLAIMER)
    st.caption("GenePrint · Public-data research demonstration")


if __name__ == "__main__":
    main()
