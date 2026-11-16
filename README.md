# Customer Analytics & Prediction Platform

## Objective

This project explores customer behavior and builds a practical churn prediction workflow. It accepts customer records, summarizes important patterns, estimates churn risk, groups customers into behavioral segments, and presents the results in an interactive dashboard and prediction API.

The goal is to help a business investigate questions such as:

- Which customer groups have higher observed churn?
- How well can a model distinguish customers who churn from those who stay?
- Which current customers have the highest estimated churn risk?
- What customer segments can be explored for retention planning?

## Features

- Upload a customer CSV or use the reproducible synthetic example dataset.
- Explore customer records with interactive charts and feature filters.
- Train and compare Logistic Regression, Random Forest, and optional XGBoost classifiers.
- Evaluate churn predictions with ROC AUC, precision, recall, F1, accuracy, confusion matrix, and ROC curve.
- Score customers with churn probability and low, medium, or high risk bands.
- Segment customers with K-means and inspect segment profiles and silhouette score.
- Serve single and batch predictions through FastAPI.
- Log CLI training runs and metrics with MLflow.

## Dataset

The bundled generator creates a synthetic, Telco-style customer dataset. It is intended to demonstrate the application workflow and does not represent real customers or measured business outcomes.

The generated data includes a `Churn` target (`Yes` or `No`) and example fields such as:

| Field | Description |
| --- | --- |
| `customerID` | Synthetic customer identifier; excluded from model features |
| `tenure` | Customer tenure in months |
| `MonthlyCharges` | Customer's monthly charge amount |
| `TotalCharges` | Approximate cumulative charges |
| `Contract` | Contract type, such as month-to-month or one/two year |
| `PaymentMethod` | Customer payment method |
| `InternetService` | Internet service type |
| `TechSupport` | Tech support service status |
| `Churn` | Target label: `Yes` means churned; `No` means retained |

For uploaded data, the dashboard looks for a `Churn`, `Exited`, or `target` label for supervised training. Labels can be `Yes`/`No` or `1`/`0`. The model can use numeric and categorical columns; common fields such as tenure and charges are normalized automatically. A CSV without a target can still be explored and clustered, but cannot be used for supervised churn evaluation.

## Data preparation

The application removes duplicate rows, trims surrounding whitespace, treats blank strings and `?` as missing values, and converts common charge and tenure fields to numeric values. Numeric model fields are imputed with the training-set median; categorical fields are imputed with the most frequent training-set value and one-hot encoded. Numeric features are standardized as part of the model pipeline.

The labeled data is split into stratified training and test sets (80/20 by default). Logistic Regression and Random Forest use class weighting to account for class imbalance. The demo does not apply over- or under-sampling. All preprocessing is fitted inside the training pipeline to avoid using holdout rows to fit the transformations.

## Exploratory analysis and dashboard

The Streamlit dashboard provides customer filters and summary metrics, then visualizes:

- Churned customer count by contract.
- Customer distribution by payment method.
- Customer tenure against monthly charges.
- Churn share by tech support and paperless billing status.
- Average monthly charges across tenure groups and churn status.
- Data preview, missing-value summary, and numeric distributions.
- Descriptive business insights based on labeled group churn rates.

Dashboard filters are generated from fields present in the uploaded dataset. Expand a filter to check or uncheck values; use **Select all** to toggle all values in that feature.

## Modeling and evaluation

The churn model is trained from the labeled customer data and evaluated on a stratified holdout set. Available algorithms are:

1. Logistic Regression
2. Random Forest
3. XGBoost (optional dependency)

The dashboard reports ROC AUC, precision, recall, F1, and accuracy, alongside a confusion matrix and ROC curve. It also scores the current customer records and allows the scored table to be downloaded as CSV. Choose the model based on the business cost of missed churners versus unnecessary retention outreach; no single metric should be treated as sufficient.

The sample project does not publish a fixed model score because results depend on the data uploaded and the generated sample. The synthetic data is useful for trying the interface, not for making real customer decisions.

## Customer segmentation

The segmentation view applies K-means to available numeric customer measures, including tenure, monthly and total charges, and senior-citizen status. If service fields are available, it also derives a count of selected services. The view shows cluster profiles, a segment map, and a silhouette score when it can be calculated. Segments are descriptive groups and should be profiled before using them to shape customer outreach.

## API

Start the API with `uvicorn api.main:app --reload`. The service exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service status and whether a model artifact exists |
| `GET /model` | Model algorithm, feature list, and saved evaluation metrics |
| `POST /predict` | Score one customer record |
| `POST /predict/batch` | Score a list of customer records |

Interactive API documentation is available at `http://localhost:8000/docs`. Example single-record request:

```json
{
  "data": {
    "tenure": 8,
    "MonthlyCharges": 89.5,
    "TotalCharges": 716,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "TechSupport": "No"
  }
}
```

The response contains a churn probability, predicted churn label, and risk band. For consistent API behavior, train and save a model using the same schema as the records being scored.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
python -m src.train
```

Run the dashboard and API in separate terminals:

```bash
streamlit run app.py
```

```bash
uvicorn api.main:app --reload
```

- Dashboard: `http://localhost:8501`
- API documentation: `http://localhost:8000/docs`
- MLflow tracking: runs locally by default when MLflow is installed; set `MLFLOW_TRACKING_URI` to use another tracking store.
- XGBoost: optional; install it with `pip install xgboost` to enable the XGBoost model choice.

## Screenshots for the portfolio README

Capture these from the running app using the generated sample data. Save them in `docs/images/` and add the Markdown image lines below to the relevant sections above. Keep the browser zoom and window size consistent so the dashboard looks cohesive.

1. **`dashboard_overview.png`** — Full dashboard overview showing the dark theme, KPI cards, expanded sidebar filters with checkbox values, and the contract/payment charts. This should be the README's lead image.
2. **`churn_analysis.png`** — Overview lower section showing churn distribution by tech support and paperless billing, plus average monthly charges by tenure and churn status.
3. **`model_evaluation.png`** — Churn Model tab with model selector, evaluation metrics, confusion matrix, and ROC curve visible together.
4. **`customer_segments.png`** — Segments tab with cluster summary, silhouette score, and segment map.
5. **`api_docs.png`** — FastAPI `/docs` page showing the prediction endpoints. Optionally expand `/predict` and show a sample response.

Suggested images in README (uncomment a line after adding the matching file):

```markdown
![Customer analytics dashboard overview](docs/images/dashboard_overview.png)
![Churn analysis charts](docs/images/churn_analysis.png)
![Churn model evaluation](docs/images/model_evaluation.png)
![Customer segments](docs/images/customer_segments.png)
![FastAPI prediction endpoints](docs/images/api_docs.png)
```

## Project structure

```text
app.py                         Streamlit dashboard
api/main.py                    FastAPI prediction endpoints
src/data.py                    Data cleanup and target-label utilities
src/modeling.py                Training, scoring, clustering, and insights
src/train.py                   CLI training and MLflow logging
scripts/generate_sample_data.py Synthetic customer data generator
data/                          Generated customer CSV (created locally)
models/                        Saved model artifact (created locally)
```

## Limitations and next steps

This is a portfolio demonstration. Before using a model for real customer decisions, validate it on representative historical data, consider time-based validation, review fairness and privacy requirements, and monitor performance and data drift. The API currently has no authentication or rate limiting.
