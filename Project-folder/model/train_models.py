import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --------------------------------------------------
# Create directories
# --------------------------------------------------
os.makedirs("model/saved_models", exist_ok=True)

# --------------------------------------------------
# Load Dataset (Local CSV)
# --------------------------------------------------
df = pd.read_csv("online_shoppers.csv")

# --------------------------------------------------
# Target Encoding
# --------------------------------------------------
df["Revenue"] = df["Revenue"].astype(int)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# --------------------------------------------------
# Encode Categorical Columns
# --------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Initialize Models
# --------------------------------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "xgboost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

# --------------------------------------------------
# Train & Save Models
# --------------------------------------------------
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(
        model,
        f"model/saved_models/{name}.pkl"
    )

# --------------------------------------------------
# Save Preprocessing Objects
# --------------------------------------------------
joblib.dump(scaler, "model/saved_models/scaler.pkl")
joblib.dump(encoders, "model/saved_models/encoders.pkl")

print("\n✅ All models and preprocessing objects saved successfully!")
