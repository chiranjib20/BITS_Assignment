import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00330/online_shoppers_intention.csv"


def load_and_preprocess(return_objects=False):
    # Load dataset from UCI
    df = pd.read_csv(UCI_URL)

    # Target
    y = df["Revenue"].astype(int)
    X = df.drop("Revenue", axis=1)

    # Encode categorical columns
    encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (internal)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    if return_objects:
        return X_train, X_test, y_train, y_test, scaler, encoders

    return X_train, X_test, y_train, y_test
