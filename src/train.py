"""Command-line trainer and optional MLflow logger."""

import argparse
import json
import pandas as pd
from .modeling import train_churn_model, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/customers.csv")
    parser.add_argument(
        "--model",
        default="Random Forest",
        choices=["Random Forest", "Logistic Regression", "XGBoost"],
    )
    args = parser.parse_args()
    result = train_churn_model(pd.read_csv(args.data), args.model)
    path = save_model(result)
    print(
        json.dumps(
            {
                "model_path": str(path),
                "algorithm": result["algorithm"],
                "metrics": result["metrics"],
            },
            indent=2,
        )
    )
    try:
        import mlflow

        with mlflow.start_run():
            mlflow.log_param("algorithm", result["algorithm"])
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in result["metrics"].items()
                    if isinstance(v, (int, float))
                }
            )
            mlflow.log_artifact(str(path))
    except ImportError:
        print("MLflow is not installed; skipped experiment logging.")


if __name__ == "__main__":
    main()
