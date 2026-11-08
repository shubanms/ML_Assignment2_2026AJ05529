"""
ML Assignment 2 - Streamlit app
Loads the 5 pre-trained models from model/ and evaluates them on an uploaded
test CSV. No retraining happens here.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classifier Comparison Lab", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}
    label_enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return models, label_enc, meta


def compute_metrics(model, X, y, n_classes):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted",
                                labels=list(range(n_classes)))
    except ValueError:  # e.g. only one class present in the uploaded file
        auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


models, label_enc, meta = load_artifacts()
target, features = meta["target"], meta["features"]
class_names = meta["classes"]
n_classes = len(class_names)

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption(f"Target column: `{target}`")
st.sidebar.caption(f"Expected features: {len(features)}")
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("Download sample test_data.csv", f,
                                   file_name="test_data.csv", mime="text/csv")

# ---------------- Main ----------------
st.title("Classifier Comparison Lab")
st.write("Upload the held-out **test CSV** (features + target column). "
         "The app scores it with the saved models - nothing is retrained.")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Waiting for a CSV upload. You can download a sample from the sidebar.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()

unknown = set(df[target].astype(str)) - set(class_names)
if unknown:
    st.error(f"Target has labels the model never saw: {sorted(unknown)}")
    st.stop()

X = df[features].copy()
for c in X.select_dtypes(include="bool").columns:  # same as training
    X[c] = X[c].astype(int)
y = label_enc.transform(df[target].astype(label_enc.classes_.dtype))

with st.expander(f"Preview uploaded data ({df.shape[0]} rows x {df.shape[1]} cols)"):
    st.dataframe(df.head(20), width="stretch")

tab_single, tab_compare = st.tabs([f"{model_name}", "Compare all 5 models"])

with tab_single:
    metrics, y_pred = compute_metrics(models[model_name], X, y, n_classes)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, y_pred, labels=list(range(n_classes)),
                                       target_names=class_names, output_dict=True,
                                       zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

with tab_compare:
    rows = []
    for name, mdl in models.items():
        m, _ = compute_metrics(mdl, X, y, n_classes)
        rows.append({"ML Model Name": name, **m})
    comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    metric_pick = st.selectbox("Metric to plot", comp.columns.tolist(), index=4)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    comp[metric_pick].sort_values().plot.barh(ax=ax2, color="#4c72b0")
    ax2.set_xlabel(metric_pick)
    ax2.set_xlim(0, 1)
    st.pyplot(fig2)
    st.success(f"Best on {metric_pick}: **{comp[metric_pick].idxmax()}**")
