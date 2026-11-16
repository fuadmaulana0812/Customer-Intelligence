import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from src.data import clean_customers, find_target, load_csv, target_to_binary
from src.modeling import (
    business_insights,
    predict_customers,
    segment_customers,
    train_churn_model,
)

st.set_page_config(page_title="Customer Intelligence", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp{background:#090D17;color:#EDF2FA;font-family:Arial,sans-serif}.block-container{padding-top:.45rem;padding-bottom:.2rem;max-width:1600px}
[data-testid="stHeader"]{background:#090D17}[data-testid="stMetric"]{background:#111827;border:1px solid #263146;padding:14px;border-radius:10px}
[data-testid="stMetricValue"]{color:#F5F7FB}.hero{padding:2px 0 8px;border-bottom:1px solid #263146;margin-bottom:10px}
.hero small{color:#45D5CD;letter-spacing:2px}.hero h1{color:#F2F5FC;font-size:25px;margin:2px 0}.hero p{color:#94A3B8;margin:0;font-size:12px}
.stPlotlyChart{margin-bottom:-1rem}h3{font-size:15px!important;margin:0 0 .15rem!important}
.insight{background:#111827;border-left:3px solid #45D5CD;border-radius:6px;padding:12px;margin:8px 0;color:#D1D9E6}
[data-testid="stSidebar"]{background:#0D1320;border-right:1px solid #20293A}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:#111827;border:1px solid #20293A;border-radius:8px;margin-bottom:7px}
[data-testid="stSidebar"] [data-testid="stMultiSelect"]{font-size:13px}
</style>""",
    unsafe_allow_html=True,
)
COLORS = ["#45D5CD", "#F28B50", "#7289F5", "#C779D0", "#F0C76A"]
PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#B8C3D4"),
    margin=dict(l=15, r=15, t=35, b=10),
    colorway=COLORS,
)


@st.cache_data
def sample():
    path = Path("data/customers.csv")
    if path.exists():
        return clean_customers(pd.read_csv(path))
    from scripts.generate_sample_data import generate

    return clean_customers(generate())


@st.cache_resource
def train(data, algorithm):
    return train_churn_model(data, algorithm)


def set_filter_values(field, value_keys):
    checked = st.session_state[f"filter_all_{field}"]
    for key in value_keys:
        st.session_state[key] = checked


def sync_filter_all(field, value_keys):
    st.session_state[f"filter_all_{field}"] = all(
        st.session_state.get(key, False) for key in value_keys
    )


st.sidebar.markdown("### ◈ CUSTOMER INTELLIGENCE")
st.sidebar.caption("Analytics · Churn risk · Segments")
upload = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
if upload:
    try:
        data = load_csv(upload)
        st.sidebar.success(f"Loaded {len(data):,} records")
    except ValueError as exc:
        st.sidebar.error(str(exc))
        data = sample()
else:
    data = sample()
    st.sidebar.info("Using included synthetic sample")

filtered = data.copy()
st.sidebar.markdown("#### FILTERS")
filter_fields = [
    ("City", "City"),
    ("Contract", "Contract"),
    ("Internet Service", "InternetService"),
    ("Payment Method", "PaymentMethod"),
    ("Tech Support", "TechSupport"),
    ("Paperless Billing", "PaperlessBilling"),
]
for label, field in filter_fields:
    if field in data:
        values = sorted(data[field].dropna().astype(str).unique())
        value_keys = [f"filter_{field}_value_{index}" for index in range(len(values))]
        for key in value_keys:
            st.session_state.setdefault(key, True)
        st.session_state.setdefault(f"filter_all_{field}", True)
        with st.sidebar.expander(label, expanded=False):
            st.checkbox(
                "Select all",
                key=f"filter_all_{field}",
                on_change=set_filter_values,
                args=(field, value_keys),
            )
            selected = []
            for value, key in zip(values, value_keys):
                st.checkbox(
                    value,
                    key=key,
                    on_change=sync_filter_all,
                    args=(field, value_keys),
                )
                if st.session_state[key]:
                    selected.append(value)
        filtered = filtered[filtered[field].astype(str).isin(selected)]
if target := find_target(data):
    values = sorted(data[target].dropna().astype(str).unique())
    value_keys = [f"filter_{target}_value_{index}" for index in range(len(values))]
    for key in value_keys:
        st.session_state.setdefault(key, True)
    st.session_state.setdefault(f"filter_all_{target}", True)
    with st.sidebar.expander("Churn Status", expanded=False):
        st.checkbox(
            "Select all",
            key=f"filter_all_{target}",
            on_change=set_filter_values,
            args=(target, value_keys),
        )
        selected = []
        for value, key in zip(values, value_keys):
            st.checkbox(
                value,
                key=key,
                on_change=sync_filter_all,
                args=(target, value_keys),
            )
            if st.session_state[key]:
                selected.append(value)
    filtered = filtered[filtered[target].astype(str).isin(selected)]
if filtered.empty:
    st.info(
        "No customers match these filters. Select at least one value in a filter to continue."
    )
    st.stop()
target = find_target(filtered)
labels = target_to_binary(filtered[target]) if target else None
rate = labels.mean() if target else None
monthly = (
    pd.to_numeric(filtered["MonthlyCharges"], errors="coerce").mean()
    if "MonthlyCharges" in filtered
    else None
)
st.markdown(
    "<div class='hero'><small>CUSTOMER INTELLIGENCE PLATFORM</small><h1>Customer Analytics & Prediction</h1><p>Understand customer behavior, identify churn risk, and find actionable segments.</p></div>",
    unsafe_allow_html=True,
)
a, b, c, d = st.columns(4)
a.metric("Total customers", f"{len(filtered):,}")
b.metric("Churned customers", f"{int(labels.sum()):,}" if target else "Add churn label")
c.metric("Churn rate", f"{rate:.1%}" if target else "—")
d.metric("Avg. monthly charges", f"${monthly:,.2f}" if pd.notna(monthly) else "—")
overview, model_tab, segment_tab, explorer = st.tabs(
    ["OVERVIEW", "CHURN MODEL", "SEGMENTS", "DATA EXPLORER"]
)

with overview:
    first, second, third = st.columns(3, gap="small")

    with first:
        st.subheader("Churn by contract")
        if target and "Contract" in filtered:
            chart = (
                filtered.assign(_churn=labels)
                .groupby("Contract", as_index=False)["_churn"]
                .sum()
            )
            fig = px.bar(
                chart, x="Contract", y="_churn", color="Contract", text_auto=True
            )
            fig.update_layout(
                **{
                    **PLOT,
                    "showlegend": False,
                    "height": 220,
                    "margin": dict(l=8, r=8, t=20, b=8),
                }
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        elif "Contract" in filtered:
            fig = px.histogram(filtered, x="Contract", color="Contract", **PLOT)
            fig.update_layout(height=220, margin=dict(l=8, r=8, t=20, b=8))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No contract field in the uploaded data.")

    with second:
        st.subheader("Customer base by payment method")
        if "PaymentMethod" in filtered:
            counts = (
                filtered["PaymentMethod"]
                .value_counts()
                .rename_axis("Payment method")
                .reset_index(name="Customers")
            )
            fig = px.pie(counts, values="Customers", names="Payment method", hole=0.62)
            fig.update_traces(textinfo="percent", textposition="inside")
            fig.update_layout(
                **{
                    **PLOT,
                    "height": 220,
                    "showlegend": True,
                    "legend": dict(orientation="h", y=-0.08, font=dict(size=9)),
                    "margin": dict(l=5, r=5, t=10, b=20),
                }
            )
            fig.add_annotation(
                text=f"{len(filtered):,}<br><sup>customers</sup>",
                showarrow=False,
                font=dict(size=16, color="white"),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No payment method field in the uploaded data.")

    with third:
        st.subheader("Monthly charges by tenure")
        if target and {"tenure", "MonthlyCharges"}.issubset(filtered.columns):
            trend = filtered[["tenure", "MonthlyCharges", target]].copy()
            trend["tenure"] = pd.to_numeric(trend["tenure"], errors="coerce")
            trend["MonthlyCharges"] = pd.to_numeric(
                trend["MonthlyCharges"], errors="coerce"
            )
            trend["Tenure group"] = (trend["tenure"] // 12 * 12).clip(upper=72)
            trend["Churn status"] = target_to_binary(trend[target]).map(
                {0: "No", 1: "Yes"}
            )
            trend = trend.dropna(subset=["MonthlyCharges", "Tenure group"])
            trend = (
                trend.groupby(["Tenure group", "Churn status"], observed=True)[
                    "MonthlyCharges"
                ]
                .mean()
                .reset_index()
            )
            fig = px.line(
                trend,
                x="Tenure group",
                y="MonthlyCharges",
                color="Churn status",
                markers=True,
                color_discrete_map={"No": "#45D5CD", "Yes": "#F28B50"},
            )
            fig.update_layout(
                **{
                    **PLOT,
                    "height": 220,
                    "yaxis_title": "Avg. monthly ($)",
                    "xaxis_title": "Tenure (months)",
                    "legend_title": None,
                    "legend": dict(orientation="h", y=1.08, font=dict(size=9)),
                    "margin": dict(l=8, r=8, t=20, b=8),
                }
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Add tenure, MonthlyCharges, and Churn to show the trend.")

    fourth, fifth, sixth = st.columns(3, gap="small")
    for container, field, heading in [
        (fourth, "TechSupport", "Churn by tech support"),
        (fifth, "PaperlessBilling", "Churn by paperless billing"),
    ]:
        with container:
            st.subheader(heading)
            if target and field in filtered:
                view = filtered[[field, target]].copy()
                view["Churn status"] = target_to_binary(view[target]).map(
                    {0: "No", 1: "Yes"}
                )
                distribution = (
                    view.groupby([field, "Churn status"], observed=True)
                    .size()
                    .rename("Customers")
                    .reset_index()
                )
                distribution["Share"] = distribution[
                    "Customers"
                ] / distribution.groupby(field)["Customers"].transform("sum")
                fig = px.bar(
                    distribution,
                    x="Share",
                    y=field,
                    color="Churn status",
                    orientation="h",
                    barmode="stack",
                    text=distribution["Share"].map(lambda value: f"{value:.0%}"),
                    color_discrete_map={"No": "#45D5CD", "Yes": "#F28B50"},
                )
                fig.update_layout(
                    **{
                        **PLOT,
                        "height": 220,
                        "xaxis_tickformat": ".0%",
                        "xaxis_range": [0, 1],
                        "xaxis_title": "Customer share",
                        "yaxis_title": None,
                        "legend": dict(orientation="h", y=1.32, font=dict(size=9)),
                        "margin": dict(l=8, r=8, t=20, b=8),
                    }
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.info(f"Add {field} and a Churn label to compare churn.")

    with sixth:
        st.subheader("Business insights")
        for insight in business_insights(filtered)[:3]:
            st.markdown(
                f"<div class='insight' style='padding:8px;margin:5px 0;font-size:12px'>- {insight}</div>",
                unsafe_allow_html=True,
            )
with model_tab:
    if not target:
        st.info(
            "A Churn field is required for supervised evaluation. Upload labeled customer data to train a model."
        )
    else:
        model_col, note_col = st.columns([1, 3], vertical_alignment="center")
        algorithm = model_col.selectbox(
            "Model", ["Random Forest", "Logistic Regression", "XGBoost"]
        )
        note_col.caption(
            "Stratified 80/20 holdout. Risk scores are estimates, not guarantees."
        )
        try:
            with st.spinner("Training model and evaluating holdout data…"):
                result = train(data, algorithm)
            metrics = result["metrics"]
            cols = st.columns(5)
            for col, label, key in zip(
                cols,
                ["ROC AUC", "Precision", "Recall", "F1 score", "Accuracy"],
                ["roc_auc", "precision", "recall", "f1", "accuracy"],
            ):
                col.metric(label, f"{metrics[key]:.3f}")
            left, right = st.columns(2)
            with left:
                fig = px.imshow(
                    metrics["confusion_matrix"],
                    text_auto=True,
                    x=["Predicted stay", "Predicted churn"],
                    y=["Actual stay", "Actual churn"],
                    color_continuous_scale=[[0, "#111827"], [1, "#45D5CD"]],
                )
                fig.update_layout(**{
                    **PLOT, "title": "Confusion matrix", "coloraxis_showscale": False,
                    "height": 250, "margin": dict(l=20, r=10, t=35, b=15),
                })
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            with right:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=metrics["roc_curve"]["fpr"],
                        y=metrics["roc_curve"]["tpr"],
                        name="ROC",
                        line=dict(color="#45D5CD", width=3),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Baseline",
                        line=dict(color="#586477", dash="dash"),
                    )
                )
                fig.update_layout(**{
                    **PLOT,
                    "title": "ROC curve",
                    "xaxis_title": "False positive rate",
                    "yaxis_title": "True positive rate",
                    "height": 250,
                    "margin": dict(l=20, r=10, t=35, b=15),
                })
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            scored = predict_customers(result, filtered).sort_values(
                "churn_probability", ascending=False
            )
            with st.expander("Customer churn risk scores", expanded=False):
                st.dataframe(scored.head(100), width="stretch", hide_index=True)
                st.download_button(
                    "Download scored customers",
                    scored.to_csv(index=False).encode(),
                    "customer_churn_predictions.csv",
                    "text/csv",
                )
        except Exception as exc:
            st.error(str(exc))

with segment_tab:
    st.subheader("Behavior-based customer groups")
    k = st.slider("Number of segments", 2, 8, 4)
    try:
        groups, silhouette = segment_customers(filtered, k)
        if target:
            groups["_churn"] = target_to_binary(groups[target])
        aggregations = {"customers": ("segment", "size")}
        for field in ["tenure", "MonthlyCharges", "TotalCharges", "_churn"]:
            if field in groups:
                aggregations[field] = (field, "mean")
        summary = (
            groups.groupby("segment")
            .agg(**aggregations)
            .reset_index()
            .rename(columns={"_churn": "churn_rate"})
        )
        left, right = st.columns([1, 2])
        left.metric(
            "Silhouette score", f"{silhouette:.3f}" if silhouette is not None else "—"
        )
        left.dataframe(summary, width="stretch", hide_index=True)
        x = "tenure" if "tenure" in groups else "MonthlyCharges"
        y = "MonthlyCharges" if "MonthlyCharges" in groups else x
        fig = px.scatter(
            groups,
            x=x,
            y=y,
            color=target if target else "segment",
            opacity=0.6,
            title="Customer segment map",
        )
        fig.update_layout(**PLOT)
        right.plotly_chart(fig, width="stretch")
        st.download_button(
            "Download segment assignments",
            groups.to_csv(index=False).encode(),
            "customer_segments.csv",
            "text/csv",
        )
        st.caption(
            "K-means groups are descriptive; review segment profiles before deciding on actions."
        )
    except Exception as exc:
        st.warning(str(exc))

with explorer:
    st.subheader("Dataset profile")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(filtered):,}")
    b.metric("Fields", len(filtered.columns))
    c.metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")
    st.dataframe(filtered.head(500), width="stretch", hide_index=True)
    numeric = filtered.select_dtypes(include="number").columns.tolist()
    if numeric:
        field = st.selectbox("Numeric distribution", numeric)
        fig = px.histogram(filtered, x=field, marginal="box")
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, width="stretch")
    missing = filtered.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        st.plotly_chart(
            px.bar(
                x=missing.values,
                y=missing.index,
                orientation="h",
                labels={"x": "Missing cells", "y": "Field"},
                **PLOT,
            ),
            width="stretch",
        )
    else:
        st.success("No missing values in this view.")

st.markdown(
    "<div style='color:#66758B;text-align:center;border-top:1px solid #263146;margin-top:28px;padding:16px;font-size:12px'>CUSTOMER ANALYTICS & PREDICTION PLATFORM · DEMO DATA IS SYNTHETIC</div>",
    unsafe_allow_html=True,
)
