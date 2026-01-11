import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest,
    xgboost_model
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Online Shoppers ML App", layout="wide")
st.title("Online Shoppers Purchasing Intention – Classification Models")

# -----------------------------
# Dataset Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Online Shoppers Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the dataset CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Preprocessing
# -----------------------------
if "Revenue" not in df.columns:
    st.error("Dataset must contain the target column 'Revenue'")
    st.stop()

y = df["Revenue"].astype(int)
X = df.drop("Revenue", axis=1)

# Encode categorical features
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Model Selection
# -----------------------------
model_map = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "K-Nearest Neighbors": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest (Ensemble)": random_forest,
    "XGBoost (Ensemble)": xgboost_model
}

selected_model = st.selectbox("Select Classification Model", model_map.keys())

# -----------------------------
# Train & Evaluate
# -----------------------------
st.subheader(f"Model Evaluation: {selected_model}")

results = model_map[selected_model].run_model(
    X_train, X_test, y_train, y_test
)

# Metrics
st.write(f"Accuracy: {results['accuracy']:.4f}")
st.write(f"AUC Score: {results['auc']:.4f}")
st.write(f"Precision: {results['precision']:.4f}")
st.write(f"Recall: {results['recall']:.4f}")
st.write(f"F1 Score: {results['f1']:.4f}")
st.write(f"MCC Score: {results['mcc']:.4f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = results["confusion_matrix"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
st.text(results["report"])
