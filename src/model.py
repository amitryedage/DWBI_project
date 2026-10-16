"""
Machine Learning and Predictive Analytics Module for Churn Modeling (PRD Stretch Goals).
Supports Logistic Regression and Random Forest with graceful fallback.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


FEATURE_COLS = [
    "age",
    "balance",
    "num_products",
    "is_active_member",
    "credit_score",
    "tenure_years",
    "has_credit_card",
    "estimated_salary",
]

FEATURE_DISPLAY_NAMES = {
    "num_products": "Number of Products",
    "is_active_member": "Active Member Status",
    "age": "Customer Age",
    "balance": "Account Balance",
    "tenure_years": "Tenure (Years)",
    "credit_score": "Credit Score",
    "estimated_salary": "Estimated Salary",
    "has_credit_card": "Credit Card Holder",
}


class PureNumpyLogisticFallback:
    """Fallback logistic model when scikit-learn is not installed."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.means = None
        self.stds = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 250, lr: float = 0.05):
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        X_scaled = (X - self.means) / self.stds

        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(epochs):
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            y_pred = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
            dw = (1 / n_samples) * np.dot(X_scaled.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= lr * dw
            self.bias -= lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - self.means) / self.stds
        linear_model = np.dot(X_scaled, self.weights) + self.bias
        prob_1 = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
        return np.column_stack([1.0 - prob_1, prob_1])


@st.cache_resource
def train_churn_model(
    df: pd.DataFrame, model_type: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Train a predictive churn model on the banking dataset.
    Returns model bundle containing the fitted model, feature importance, and performance.
    """
    X = df[FEATURE_COLS].copy()
    y = df["churned"].values

    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, random_state=4010)
            clf.fit(X_scaled, y)
            importances = np.abs(clf.coef_[0])
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=4010)
            clf.fit(X, y)
            importances = clf.feature_importances_

        accuracy = float(clf.score(X if model_type == "Random Forest" else X_scaled, y))
    else:
        scaler = None
        clf = PureNumpyLogisticFallback()
        clf.fit(X.values, y)
        importances = np.abs(clf.weights)
        preds = (clf.predict_proba(X.values)[:, 1] >= 0.5).astype(int)
        accuracy = float((preds == y).mean())

    importance_df = pd.DataFrame({
        "feature": [FEATURE_DISPLAY_NAMES.get(col, col) for col in FEATURE_COLS],
        "feature_key": FEATURE_COLS,
        "importance": importances,
    }).sort_values(by="importance", ascending=True)

    # Normalize importances to percentages
    total_imp = importance_df["importance"].sum()
    if total_imp > 0:
        importance_df["importance_pct"] = (importance_df["importance"] / total_imp) * 100.0
    else:
        importance_df["importance_pct"] = 0.0

    return {
        "model": clf,
        "scaler": scaler if SKLEARN_AVAILABLE else None,
        "model_type": model_type,
        "accuracy": accuracy,
        "importance_df": importance_df,
        "sklearn_used": SKLEARN_AVAILABLE,
    }


def predict_single_customer_churn(
    model_bundle: Dict[str, Any], customer_data: Dict[str, Any]
) -> float:
    """
    Predict churn probability (0.0 to 100.0%) for a given set of customer attributes.
    """
    clf = model_bundle["model"]
    scaler = model_bundle["scaler"]
    model_type = model_bundle["model_type"]

    row_values = np.array([[customer_data[col] for col in FEATURE_COLS]], dtype=float)

    if model_bundle["sklearn_used"]:
        if model_type == "Logistic Regression" and scaler is not None:
            row_scaled = scaler.transform(row_values)
            proba = float(clf.predict_proba(row_scaled)[0, 1])
        else:
            proba = float(clf.predict_proba(row_values)[0, 1])
    else:
        proba = float(clf.predict_proba(row_values)[0, 1])

    return proba * 100.0
