import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

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

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Online Shoppers – Pretrained ML Models",
    layout="wide"
)

st.title("Online Shoppers Purchasing Intention – Pretrained Models")

# --------------------------------------------------
# Load Pretrained Objects
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("Project-folder/model/saved_models/logistic.pkl"),
        "Decision Tree": joblib.load("Project-folder/model/saved_models/decision_tree.pkl"),
        "KNN": joblib.load("Project-folder/model/saved_models/knn.pkl"),
        "Naive Bayes": joblib.load("Project-folder/model/saved_models/naive_bayes.pkl"),
        "Random Forest (Ensemble)": joblib.load("Project-folder/model/saved_models/random_forest.pkl"),
        "XGBoost (Ensemble)": joblib.load("Project-folder/model/saved_models/xgboost.pkl")
    }

    scaler = joblib.load("Project-folder/model/saved_models/scaler.pkl")
    encoders = joblib.load("Project-folder/model/saved_models/encoders.pkl")

    return models, scaler, encoders


models, scaler, encoders = load_artifacts()

# --------------------------------------------------
# Upload Test Dataset
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST dataset (CSV only)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a test dataset to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Validate Dataset
# --------------------------------------------------
if "Revenue" not in df.columns:
    st.error("Uploaded dataset must contain the target column 'Revenue'")
    st.stop()

y_test = df["Revenue"].astype(int)
X_test = df.drop("Revenue", axis=1)

# --------------------------------------------------
# Encode Categorical Features (Using TRAIN Encoders)
# --------------------------------------------------
for col, encoder in encoders.items():
    if col in X_test.columns:
        X_test[col] = encoder.transform(X_test[col])

# --------------------------------------------------
# Scale Features (Using TRAIN Scaler)
# --------------------------------------------------
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
selected_model_name = st.selectbox(
    "Select Pretrained Model",
    list(models.keys())
)

model = models[selected_model_name]

# --------------------------------------------------
# Prediction & Evaluation
# --------------------------------------------------
try:
    y_pred = model.predict(X_test)
except Exception as e:
    st.warning("Standard prediction failed. Using safe XGBoost prediction.")
    y_pred = model.predict(X_test, validate_features=False)

if hasattr(model, "predict_proba"):
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None
else:
    y_prob = None

st.subheader(f"Evaluation Metrics – {selected_model_name}")

st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
st.write(f"MCC Score: {matthews_corrcoef(y_test, y_pred):.4f}")

if y_prob is not None:
    st.write(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
