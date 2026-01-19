# evaluate_models.py

import joblib
import pandas as pd
import xgboost as xgb
from xgboost import Booster
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

MODEL_DIR = "Project-folder/model/saved_models"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("online_shoppers.csv")

# Load preprocessing objects
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")

# Encode categorical columns
for col, le in encoders.items():
    df[col] = le.transform(df[col])

# Features & target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Use SAME test split as training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Load sklearn models
# -----------------------------
models = {
    "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic.pkl"),
    "Decision Tree": joblib.load(f"{MODEL_DIR}/decision_tree.pkl"),
    "KNN": joblib.load(f"{MODEL_DIR}/knn.pkl"),
    "Naive Bayes": joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
    "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
}

# Load XGBoost Booster
xgb_booster = Booster()
xgb_booster.load_model(f"{MODEL_DIR}/xgboost.json")

print("\nMODEL PERFORMANCE SCORES\n")

# -----------------------------
# Evaluate sklearn models
# -----------------------------
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"{name}")
    print(" Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print(" AUC      :", round(roc_auc_score(y_test, y_proba), 4))
    print(" Precision:", round(precision_score(y_test, y_pred), 4))
    print(" Recall   :", round(recall_score(y_test, y_pred), 4))
    print(" F1 Score :", round(f1_score(y_test, y_pred), 4))
    print(" MCC      :", round(matthews_corrcoef(y_test, y_pred), 4))
    print("-" * 40)

# -----------------------------
# Evaluate XGBoost
# -----------------------------
dtest = xgb.DMatrix(X_test_scaled)
y_proba = xgb_booster.predict(dtest)
y_pred = (y_proba >= 0.5).astype(int)

print("XGBoost")
print(" Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print(" AUC      :", round(roc_auc_score(y_test, y_proba), 4))
print(" Precision:", round(precision_score(y_test, y_pred), 4))
print(" Recall   :", round(recall_score(y_test, y_pred), 4))
print(" F1 Score :", round(f1_score(y_test, y_pred), 4))
print(" MCC      :", round(matthews_corrcoef(y_test, y_pred), 4))
print("-" * 40)
