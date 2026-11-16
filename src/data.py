"""Data loading, normalization, and churn label utilities."""

import pandas as pd

ALIASES = {
    "monthlycharges": "MonthlyCharges",
    "monthly_charges": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "total_charges": "TotalCharges",
    "customerid": "customerID",
    "customer_id": "customerID",
    "churn_status": "Churn",
    "churn": "Churn",
    "exited": "Churn",
}


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The dataset has no rows.")
    out = df.copy()
    out = out.rename(
        columns={
            c: ALIASES[c.strip().lower()]
            for c in out.columns
            if c.strip().lower() in ALIASES
        }
    )
    out = out.drop_duplicates().reset_index(drop=True)
    for col in out.select_dtypes(include="object").columns:
        out[col] = (
            out[col].astype("string").str.strip().replace({"": pd.NA, "?": pd.NA})
        )
    for col in ("TotalCharges", "MonthlyCharges", "tenure"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def find_target(df: pd.DataFrame):
    return next((c for c in ("Churn", "Exited", "target") if c in df), None)


def target_to_binary(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if set(numeric.dropna().unique()).issubset({0, 1}):
            return numeric.astype("Int64")
    labels = values.astype("string").str.strip().str.lower()
    result = labels.map(
        {
            "yes": 1,
            "true": 1,
            "1": 1,
            "churned": 1,
            "no": 0,
            "false": 0,
            "0": 0,
            "retained": 0,
        }
    )
    if result.isna().any():
        bad = labels[result.isna()].dropna().unique().tolist()[:5]
        raise ValueError(
            f"Churn labels must be Yes/No or 1/0. Unrecognized values: {bad}"
        )
    return result.astype(int)


def load_csv(file) -> pd.DataFrame:
    try:
        return clean_customers(pd.read_csv(file))
    except Exception as exc:
        raise ValueError(f"Could not read customer CSV: {exc}") from exc


def basic_profile(df: pd.DataFrame) -> dict:
    target = find_target(df)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "churn_rate": float(target_to_binary(df[target]).mean()) if target else None,
    }
