try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    xgb_error = str(e)

from sklearn.metrics import *

def run_model(X_train, X_test, y_train, y_test):
    if XGBClassifier is None:
        raise RuntimeError(
            "XGBoost could not be loaded. Error: " + xgb_error
        )

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
