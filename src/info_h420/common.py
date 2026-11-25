"""
Shared pipeline for the INFO-H420 Responsible AI project (Adult dataset).

Every section of the project imports this module so that all classifiers
(classifier, fair, private, private+fair) use exactly the same
preprocessing, the same train/test split and the same model.

Conventions
-----------
* Protected attributes are binary columns of the preprocessed DataFrame:
    age = 1 if age >= AGE_THRESHOLD (older group), else 0
    sex = 1 for Male, 0 for Female
  The value 1 is the "privileged" group (higher rate of income >50K),
  which matches the AIF360 convention.
* The label column is "income": 1 for ">50K", 0 for "<=50K".
* The split is stratified on the label only. Because noisy (LDP) versions
  of age/sex do not change the label nor the row order, calling
  split_data() on the private DataFrame gives exactly the same rows in
  train/test as on the real DataFrame (requirement of Q4).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration (change here, not in the notebooks)
# ---------------------------------------------------------------------------
SEED = 42
AGE_THRESHOLD = 37          # median age of the cleaned data, two groups of ~equal size
TEST_SIZE = 0.20
MODEL = "logreg"            # "logreg" (default) or "hgb"
LOGREG_C = 1.0              # default when no tuned value is supplied

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

LABEL = "income"
PROTECTED = ["age", "sex"]
CATEGORICAL = [
    "workclass", "marital-status", "occupation", "relationship",
    "race", "native-country",
]
NUMERICAL = ["education-num", "capital-gain", "capital-loss", "hours-per-week"]
FEATURES = PROTECTED + CATEGORICAL + NUMERICAL


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_raw(path=DATA_PATH):
    """Read adult.data as downloaded from UCI ('?' marks missing values)."""
    return pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,   # values are written as ", State-gov"
        na_values="?",
    )


def preprocess(raw, age_threshold=AGE_THRESHOLD):
    """Clean the raw data and binarize the protected attributes.

    Steps: drop rows with missing values, drop fnlwgt (census sampling
    weight, not a property of the person) and education (redundant with
    education-num), binarize age and sex, encode the label as 0/1.
    The returned DataFrame has a fresh 0..n-1 index; keep that row order
    when creating private copies of the data.
    """
    df = raw.dropna().copy()
    df = df.drop(columns=["fnlwgt", "education"])
    df["age"] = (df["age"] >= age_threshold).astype(int)
    df["sex"] = (df["sex"] == "Male").astype(int)
    df[LABEL] = (df[LABEL] == ">50K").astype(int)
    df = df.reset_index(drop=True)
    return df[FEATURES + [LABEL]]


def load_data(path=DATA_PATH, age_threshold=AGE_THRESHOLD):
    """Shortcut: load_raw + preprocess."""
    return preprocess(load_raw(path), age_threshold)


def split_indices(df, seed=SEED):
    """Return train/test row indices with an 80/20 split stratified on income."""
    return train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=df[LABEL], random_state=seed
    )


def split_data(df, seed=SEED):
    """Return train/test DataFrames using split_indices()."""
    idx_train, idx_test = split_indices(df, seed)
    return df.iloc[idx_train], df.iloc[idx_test]


def X_y(df):
    """Features and label of a (split of the) preprocessed DataFrame."""
    return df[FEATURES], df[LABEL]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_encoder():
    """One-hot for categorical columns, scaling for numerical ones,
    protected attributes (already 0/1) passed through unchanged."""
    return ColumnTransformer(
        [
            ("protected", "passthrough", PROTECTED),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL),
            ("num", StandardScaler(), NUMERICAL),
        ],
        verbose_feature_names_out=False,
    )


def make_model(model=MODEL, C=LOGREG_C, seed=SEED):
    """Unfitted pipeline: encoder + classifier."""
    if model == "logreg":
        clf = LogisticRegression(C=C, max_iter=2000)
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}")
    return Pipeline([("encoder", build_encoder()), ("clf", clf)])


def train_classifier(train, sample_weight=None, model=MODEL, C=LOGREG_C):
    """Fit the classifier on a training DataFrame.

    sample_weight: optional per-row weights, e.g. the instance weights
    produced by AIF360 Reweighing (Q3, Q5).
    """
    pipe = make_model(model, C)
    X, y = X_y(train)
    fit_params = {} if sample_weight is None else {"clf__sample_weight": sample_weight}
    pipe.fit(X, y, **fit_params)
    return pipe


def feature_names(pipe):
    """Names of the encoded features (useful for SHAP/LIME in Q6)."""
    return list(pipe.named_steps["encoder"].get_feature_names_out())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(pipe, df):
    """Standard performance metrics of a fitted pipeline on a DataFrame."""
    X, y = X_y(df)
    y_pred = pipe.predict(X)
    y_score = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_score),
    }


def evaluate_by_group(pipe, df, groups=PROTECTED):
    """Disaggregated metrics: one row per value of each protected attribute.

    Note: for the private classifier, pass a DataFrame holding the REAL
    values of age/sex if you want the groups defined by the real values.
    """
    X, y = X_y(df)
    y_pred = pd.Series(pipe.predict(X), index=df.index)
    rows = []
    for g in groups:
        for v in (0, 1):
            m = df[g] == v
            rows.append({
                "attribute": g,
                "value": v,
                "n": int(m.sum()),
                "base_rate": y[m].mean(),
                "positive_rate": y_pred[m].mean(),
                "accuracy": accuracy_score(y[m], y_pred[m]),
                "recall (TPR)": recall_score(y[m], y_pred[m]),
            })
    return pd.DataFrame(rows)
