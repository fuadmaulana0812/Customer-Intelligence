"""Generate reproducible synthetic Telco-like customer records."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate(n=2500, seed=21):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.56, 0.24, 0.20]
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.43, 0.23])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.18, 0.24, 0.24],
    )
    tech = np.where(
        internet == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    )
    monthly = np.clip(
        rng.normal(
            np.where(
                internet == "Fiber optic", 82, np.where(internet == "DSL", 58, 24)
            ),
            13,
        ),
        18,
        125,
    ).round(2)
    logit = (
        -2.05
        + (contract == "Month-to-month") * 1.25
        + (payment == "Electronic check") * 0.48
        + (internet == "Fiber optic") * 0.4
        + (tech == "No") * 0.35
        - tenure * 0.035
        + (monthly > 85) * 0.3
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "customerID": [f"C{100000+i}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.binomial(1, 0.16, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(
                ["Yes", "No", "No phone service"], n, p=[0.45, 0.45, 0.1]
            ),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["Yes", "No"], n),
            "OnlineBackup": rng.choice(["Yes", "No"], n),
            "DeviceProtection": rng.choice(["Yes", "No"], n),
            "TechSupport": tech,
            "StreamingTV": rng.choice(["Yes", "No"], n),
            "StreamingMovies": rng.choice(["Yes", "No"], n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": (monthly * tenure + rng.normal(0, 45, n)).clip(0).round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


if __name__ == "__main__":
    path = Path("data/customers.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(path, index=False)
    print(f"Generated {path} with 2500 synthetic customers.")
