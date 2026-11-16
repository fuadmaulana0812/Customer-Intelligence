"""FastAPI churn scoring service."""

from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.modeling import load_model, predict_customers, save_model, train_churn_model

app = FastAPI(title="Customer Analytics API", version="1.0.0")
MODEL_PATH = Path("models/churn_model.joblib")
_model = None


class CustomerRecord(BaseModel):
    data: dict[str, Any]


class BatchRequest(BaseModel):
    customers: list[dict[str, Any]]


def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
        if _model is None:
            data_path = Path("data/customers.csv")
            if not data_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail="No model found. Run python -m src.train first.",
                )
            try:
                _model = train_churn_model(pd.read_csv(data_path))
                save_model(_model, MODEL_PATH)
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"Unable to train model: {exc}"
                ) from exc
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None or MODEL_PATH.exists()}


@app.get("/model")
def model_info():
    model = get_model()
    return {
        "algorithm": model["algorithm"],
        "features": model["features"],
        "metrics": model["metrics"],
    }


@app.post("/predict")
def predict(record: CustomerRecord):
    model = get_model()
    try:
        row = predict_customers(model, pd.DataFrame([record.data])).iloc[0]
        return {
            "churn_probability": float(row.churn_probability),
            "predicted_churn": row.predicted_churn,
            "risk_band": str(row.risk_band),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Prediction failed: {exc}"
        ) from exc


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if not request.customers:
        raise HTTPException(
            status_code=422, detail="Provide at least one customer record."
        )
    try:
        result = predict_customers(get_model(), pd.DataFrame(request.customers))
        return {
            "predictions": result[
                ["churn_probability", "predicted_churn", "risk_band"]
            ].to_dict(orient="records")
        }
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Prediction failed: {exc}"
        ) from exc
