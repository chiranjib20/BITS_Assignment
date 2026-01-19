# streamlit_app.py

import joblib
import streamlit as st
import pandas as pd
import xgboost as xgb
from xgboost import Booster
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Online Shoppers Classification",
    layout="wide"
)

st.title("🛒 Online Shoppers Purchase Prediction - Classification Models")

MODEL_DIR = "Project-folder/model/saved_models"

# -----------------------------
# Load trained artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic.pkl"),
        "Decision Tree": joblib.load(f"{MODEL_DIR}/decision_tree.pkl"),
        "KNN": joblib.load(f"{MODEL_DIR}/knn.pkl"),
        "Naive Bayes": joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
        "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
    }

    booster = Booster()
    booster.load_model(f"{MODEL_DIR}/xgboost.json")

    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")

    return models, booster, scaler, encoders


models, xgb_booster, scaler, encoders = load_artifacts()

# -----------------------------
# Upload TEST data only
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload TEST CSV (must include Revenue column)",
    type=["csv"]
)

model_choice = st.selectbox(
    "Select Model",
    list(models.keys()) + ["XGBoost"]
)

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df_test.columns:
            df_test[col] = le.transform(df_test[col])

    X_test = df_test.drop("Revenue", axis=1)
    y_true = df_test["Revenue"]

    X_scaled = scaler.transform(X_test)

    # -----------------------------
    # Prediction
    # -----------------------------
    if model_choice == "XGBoost":
        dtest = xgb.DMatrix(X_scaled)
        y_proba = xgb_booster.predict(dtest)
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        model = models[model_choice]
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col1.metric("Precision", round(precision_score(y_true, y_pred), 4))

    col2.metric("Recall", round(recall_score(y_true, y_pred), 4))
    col2.metric("F1 Score", round(f1_score(y_true, y_pred), 4))

    col3.metric("AUC", round(roc_auc_score(y_true, y_proba), 4))
    col3.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 4))

    #st.subheader("📉 Confusion Matrix")
    #st.write(confusion_matrix(y_true, y_pred))

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    #st.subheader("Classification Report")
    #st.text(classification_report(y_true, y_pred))
