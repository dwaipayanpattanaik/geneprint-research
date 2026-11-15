# GenePrint — compact gene-expression research prototype

GenePrint demonstrates tumor-type classification among five existing tumor classes using a public 801-sample, 20,531-feature RNA-sequencing dataset. A fixed research protocol compares majority baseline, full logistic regression, and compact 10/25/50/100/500-feature alternatives. The saved model and evaluation artifacts in this repository are from the completed local experiments; no retraining occurs when users open the app.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Data and scope

Source: [UCI Gene Expression Cancer RNA-Seq, dataset 401](https://archive.ics.uci.edu/dataset/401/gene), DOI [10.24432/C5R88H](https://doi.org/10.24432/C5R88H). Research classes: BRCA, COAD, KIRC, LUAD, PRAD. The app ships only public demonstration samples; it does not include the complete research expression matrix. The saved feature-to-gene annotation is provisional. This is not a clinical diagnostic application.

The `src/` modules document training/evaluation logic, and `models/` and `results/` contain the required inference artifacts and measured outputs. The public interface is in `app.py`.
