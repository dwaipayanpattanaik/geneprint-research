"""Training-only preprocessing and exact compact inference pipeline export."""
import copy
import warnings
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import SEED


def safe_f_classif(X, y):
    """Rank training features with ANOVA; constants receive zero scores."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores, pvalues = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max), np.nan_to_num(pvalues, nan=1.0)


def make_pipeline(k="all"):
    """Each fit learns imputation, selection, scaling, then regularized logistic regression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("selector", SelectKBest(score_func=safe_f_classif, k=k)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
                                          tol=1e-4, random_state=SEED)),
    ])


def compact_pipeline(fitted, X_train):
    """Reuse fitted components, projecting imputation statistics to selected inputs.

    No fitting or missing-feature fabrication occurs during export. The resulting
    pipeline requires exactly the selected raw columns, in selector order.
    """
    selected = fitted.named_steps["selector"].get_support(indices=True)
    imputer = copy.deepcopy(fitted.named_steps["imputer"])
    imputer.statistics_ = imputer.statistics_[selected].copy()
    imputer.n_features_in_ = len(selected)
    # fit used a NumPy array, so there is no feature_names_in_ to project.
    compact = Pipeline([
        ("imputer", imputer),
        ("scaler", copy.deepcopy(fitted.named_steps["scaler"])),
        ("classifier", copy.deepcopy(fitted.named_steps["classifier"])),
    ])
    original = fitted.predict_proba(X_train)
    projected = compact.predict_proba(X_train[:, selected])
    if not np.allclose(original, projected, rtol=1e-12, atol=1e-12):
        raise ValueError("Compact export did not preserve research predictions.")
    return compact, selected

