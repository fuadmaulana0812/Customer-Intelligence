"""Churn model training, evaluation, scoring, and customer segmentation."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import clean_customers, find_target, target_to_binary


def feature_frame(df):
    x = df.drop(
        columns=[
            c
            for c in (
                "customerID",
                "CustomerID",
                "customer_id",
                "Churn",
                "Exited",
                "target",
            )
            if c in df
        ]
    ).copy()
    for col in ("TotalCharges", "MonthlyCharges", "tenure"):
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x


def build_pipeline(x, algorithm="Random Forest"):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        parts.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    prep = ColumnTransformer(parts)
    if algorithm == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif algorithm == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError(
                "Install optional XGBoost with: pip install xgboost"
            ) from exc
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def train_churn_model(df, algorithm="Random Forest", test_size=0.2):
    data = clean_customers(df)
    target = find_target(data)
    if not target:
        raise ValueError("A Churn column is required for supervised training.")
    y = target_to_binary(data[target]).astype(int)
    x = feature_frame(data)
    if y.nunique() < 2:
        raise ValueError("Training requires examples from both churn classes.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y
    )
    pipeline = build_pipeline(x_train, algorithm).fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "test_rows": len(y_test),
    }
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "features": x.columns.tolist(),
        "algorithm": algorithm,
    }


def save_model(result, path="models/churn_model.joblib"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    return path


def load_model(path="models/churn_model.joblib"):
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def predict_customers(model, df):
    x = feature_frame(clean_customers(df))
    probability = model["pipeline"].predict_proba(x)[:, 1]
    out = df.copy()
    out["churn_probability"] = probability
    out["predicted_churn"] = np.where(probability >= 0.5, "Yes", "No")
    out["risk_band"] = pd.cut(
        probability, [-0.001, 0.35, 0.65, 1.001], labels=["Low", "Medium", "High"]
    )
    return out


def segment_customers(df, clusters=4):
    data = clean_customers(df)
    cols = [
        c
        for c in ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")
        if c in data
    ]
    if not cols:
        raise ValueError(
            "Segmentation needs a numeric field such as tenure or MonthlyCharges."
        )
    features = data[cols].apply(pd.to_numeric, errors="coerce")
    if "TotalCharges" not in features and {"tenure", "MonthlyCharges"}.issubset(
        features
    ):
        features["TotalCharges"] = features.tenure * features.MonthlyCharges
    services = [
        c
        for c in (
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        )
        if c in data
    ]
    if services:
        features["service_count"] = (
            data[services]
            .apply(
                lambda c: c.astype("string")
                .str.lower()
                .isin(["yes", "1", "true"])
                .astype(int)
            )
            .sum(axis=1)
        )
    features = features.fillna(features.median()).fillna(0)
    k = min(max(2, int(clusters)), len(data) - 1)
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
    score = (
        float(silhouette_score(scaled, labels))
        if len(set(labels)) > 1 and len(data) > k
        else None
    )
    out = data.copy()
    out["segment"] = labels.astype(str)
    return out, score


def business_insights(df):
    target = find_target(df)
    if not target:
        return [
            "Add a Churn label to explore churn drivers. Customers can still be profiled and segmented."
        ]
    y = target_to_binary(df[target])
    overall = y.mean()
    insights = []
    for field in (
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "PaperlessBilling",
    ):
        if field in df and df[field].nunique() > 1:
            rates = (
                pd.DataFrame({"group": df[field].astype("string"), "churn": y})
                .groupby("group")["churn"]
                .mean()
                .sort_values(ascending=False)
            )
            insights.append(
                f"{rates.index[0]} has the highest observed churn for {field} ({rates.iloc[0]:.0%} vs {overall:.0%} overall)."
            )
    if "tenure" in df:
        tenure = pd.to_numeric(df.tenure, errors="coerce")
        insights.append(
            f"Churn is {y[tenure <= tenure.quantile(.25)].mean():.0%} among newer and {y[tenure >= tenure.quantile(.75)].mean():.0%} among longer-tenure customers."
        )
    return insights[:5] or [
        "Not enough category variation to identify churn differences yet."
    ]
