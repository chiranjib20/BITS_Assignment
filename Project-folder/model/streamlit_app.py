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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Online Shoppers ML Models", layout="wide")

st.title("Online Shoppers Purchasing Intention – Classification Models")

# -------------------------------
# Load dataset directly from UCI
# -------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    return pd.read_csv(url)

df = load_data()
st.success("Dataset loaded directly from UCI Machine Learning Repository")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

model_name = st.selectbox("Select ML Model", list(models.keys()))
model = models[model_name]

# -------------------------------
# Train & Predict
# -------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col1.metric("AUC", f"{auc:.4f}")

col2.metric("Precision", f"{precision:.4f}")
col2.metric("Recall", f"{recall:.4f}")

col3.metric("F1 Score", f"{f1:.4f}")
col3.metric("MCC", f"{mcc:.4f}")

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# Classification Report
# -------------------------------
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
