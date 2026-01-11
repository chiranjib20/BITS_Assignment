import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.data_preprocessing import load_and_preprocess
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
# Load & Train on FULL UCI DATA
# -----------------------------
X_train, X_internal_test, y_train, y_internal_test, scaler, encoders = load_and_preprocess(return_objects=True)

model_map = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "KNN": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest": random_forest,
    "XGBoost": xgboost_model
}

selected_model = st.selectbox("Select Model", model_map.keys())

# Train selected model
results = model_map[selected_model].run_model(
    X_train, X_internal_test, y_train, y_internal_test
)

model = results["model"]

st.subheader("Model Performance on Internal Test Set (UCI)")
st.write(f"Accuracy: {results['accuracy']:.4f}")
st.write(f"AUC: {results['auc']:.4f}")
st.write(f"Precision: {results['precision']:.4f}")
st.write(f"Recall: {results['recall']:.4f}")
st.write(f"F1 Score: {results['f1']:.4f}")
st.write(f"MCC: {results['mcc']:.4f}")

# -----------------------------
# Upload TEST DATA ONLY
# -----------------------------
st.markdown("---")
st.header("Upload External Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file containing ONLY test data",
    type=["csv"]
)

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    # Encode categorical columns using TRAIN encoders
    for col, encoder in encoders.items():
        if col in test_df.columns:
            test_df[col] = encoder.transform(test_df[col])

    # Separate target if exists
    y_test_external = None
    if "Revenue" in test_df.columns:
        y_test_external = test_df["Revenue"]
        X_test_external = test_df.drop("Revenue", axis=1)
    else:
        X_test_external = test_df

    # Scale using TRAIN scaler
    X_test_external = scaler.transform(X_test_external)

    # Predictions
    y_pred = model.predict(X_test_external)
    y_prob = model.predict_proba(X_test_external)[:, 1]

    st.subheader("Predictions on Uploaded Test Data")
    st.write(pd.DataFrame({
        "Prediction": y_pred,
        "Probability": y_prob
    }).head())

    # If ground truth exists → show metrics
    if y_test_external is not None:
        st.subheader("Evaluation on Uploaded Test Data")

        st.write(f"Accuracy: {accuracy_score(y_test_external, y_pred):.4f}")
        st.write(f"AUC: {roc_auc_score(y_test_external, y_prob):.4f}")
        st.write(f"Precision: {precision_score(y_test_external, y_pred):.4f}")
        st.write(f"Recall: {recall_score(y_test_external, y_pred):.4f}")
        st.write(f"F1 Score: {f1_score(y_test_external, y_pred):.4f}")
        st.write(f"MCC: {matthews_corrcoef(y_test_external, y_pred):.4f}")

        cm = confusion_matrix(y_test_external, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.text(classification_report(y_test_external, y_pred))
