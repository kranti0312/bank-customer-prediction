import io
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="BankPredict AI", page_icon="🏦", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
MODEL_FILE = Path("bank_term_deposit_model.pkl")

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1250px;}
.hero {padding: 1.35rem 1.5rem; border-radius: 16px; background: linear-gradient(115deg,#102b4e,#245b83); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#dbeafe; margin:.35rem 0 0 0;}
div[data-testid="stMetric"] {background:#f4f7fb; padding:14px; border-radius:12px; border:1px solid #e5eaf1;}
.result-yes {padding:18px; border-radius:12px; background:#e9f8ef; border:1px solid #a9dfba; color:#14532d;}
.result-no {padding:18px; border-radius:12px; background:#fff1f0; border:1px solid #f2b8b5; color:#842029;}
.small-note {font-size:.85rem; color:#64748b;}
</style>
<div class="hero">
  <h1>🏦 BankPredict AI</h1>
  <p>Bank customer term-deposit subscription prediction using machine learning</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Downloading the UCI Bank Marketing dataset...")
def load_dataset():
    response = requests.get(DATA_URL, timeout=90)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as outer:
        nested_name = next(n for n in outer.namelist() if n.endswith("bank-additional.zip"))
        with outer.open(nested_name) as nested_bytes:
            with zipfile.ZipFile(io.BytesIO(nested_bytes.read())) as inner:
                csv_name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
                with inner.open(csv_name) as csv_file:
                    data = pd.read_csv(csv_file, sep=";")
    return data

@st.cache_resource(show_spinner="Training and preparing the machine-learning models...")
def train_models(data):
    target = "y"
    X = data.drop(columns=[target, "duration"])
    y = (data[target] == "yes").astype(int)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    fitted, rows = {}, []
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, prob)
        })

    metrics = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    chosen_name = metrics.loc[0, "Model"]
    chosen_model = fitted[chosen_name]
    chosen_pred = chosen_model.predict(X_test)
    cm = confusion_matrix(y_test, chosen_pred)
    return {
        "model": chosen_model,
        "model_name": chosen_name,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "confusion_matrix": cm,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "training_note": "duration excluded to avoid data leakage"
    }

with st.spinner("Loading dataset and preparing prediction model. First run may take a few minutes..."):
    try:
        df = load_dataset()
        bundle = train_models(df)
    except Exception as exc:
        st.error("The dataset or model could not be loaded. Check your internet connection and try again.")
        st.exception(exc)
        st.stop()

model = bundle["model"]
features = bundle["feature_columns"]
metrics = bundle["metrics"]

# Sidebar
st.sidebar.title("BankPredict AI")
page = st.sidebar.radio("Navigate", ["Dashboard", "Customer Prediction", "About Project"])
st.sidebar.caption("UCI Bank Marketing dataset")
st.sidebar.caption("Model training follows the uploaded project notebook.")

if page == "Dashboard":
    st.title("Banking Analytics Dashboard")
    st.caption("Overview of customer subscription patterns and model performance.")
    total = len(df)
    yes_count = int((df["y"] == "yes").sum())
    no_count = int((df["y"] == "no").sum())
    rate = yes_count / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Subscribed", f"{yes_count:,}")
    c3.metric("Not Subscribed", f"{no_count:,}")
    c4.metric("Subscription Rate", f"{rate:.2f}%")

    left, right = st.columns(2)
    with left:
        counts = df["y"].value_counts().rename_axis("Subscription").reset_index(name="Customers")
        st.plotly_chart(px.pie(counts, names="Subscription", values="Customers",
                               hole=.45, title="Customer Subscription Distribution"),
                        use_container_width=True)
    with right:
        job = (df.assign(subscribed=(df["y"] == "yes").astype(int))
                 .groupby("job", as_index=False)["subscribed"].mean())
        job["Subscription Rate (%)"] = job["subscribed"] * 100
        job = job.sort_values("Subscription Rate (%)", ascending=False)
        st.plotly_chart(px.bar(job, x="job", y="Subscription Rate (%)",
                               title="Subscription Rate by Job", text_auto=".1f"),
                        use_container_width=True)

    month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    month = (df.assign(subscribed=(df["y"] == "yes").astype(int))
               .groupby("month", as_index=False)["subscribed"].mean())
    month["Subscription Rate (%)"] = month["subscribed"] * 100
    month["month"] = pd.Categorical(month["month"], categories=month_order, ordered=True)
    month = month.sort_values("month")
    st.plotly_chart(px.line(month, x="month", y="Subscription Rate (%)", markers=True,
                             title="Subscription Rate by Month"), use_container_width=True)

    st.subheader("Model Performance Comparison")
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "Model"}),
                 use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(metrics.melt(id_vars="Model",
                      value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                      var_name="Metric", value_name="Score"),
                      x="Model", y="Score", color="Metric", barmode="group",
                      range_y=[0, 1], title="Model Evaluation"), use_container_width=True)
    st.info(f"Selected model: **{bundle['model_name']}** (highest test F1-score). "
            "The field `duration` is excluded because it is only known after a call and can cause data leakage.")

elif page == "Customer Prediction":
    st.title("Predict a Customer")
    st.caption("Enter customer details below. The trained model will estimate the likelihood of term-deposit subscription.")
    with st.form("customer_form"):
        st.subheader("Personal Information")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            job = st.selectbox("Job", sorted(df["job"].dropna().unique()))
        with b:
            marital = st.selectbox("Marital Status", sorted(df["marital"].dropna().unique()))
            education = st.selectbox("Education", sorted(df["education"].dropna().unique()))
        with c:
            default = st.selectbox("Credit in Default", sorted(df["default"].dropna().unique()))
            housing = st.selectbox("Housing Loan", sorted(df["housing"].dropna().unique()))

        st.subheader("Loan and Campaign Information")
        a, b, c = st.columns(3)
        with a:
            loan = st.selectbox("Personal Loan", sorted(df["loan"].dropna().unique()))
            contact = st.selectbox("Contact Type", sorted(df["contact"].dropna().unique()))
            month = st.selectbox("Last Contact Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], index=4)
        with b:
            day_of_week = st.selectbox("Last Contact Day", ["mon","tue","wed","thu","fri"], index=1)
            campaign = st.number_input("Contacts During This Campaign", min_value=1, max_value=100, value=1)
            pdays = st.number_input("Days Since Previous Contact (999 = not previously contacted)",
                                    min_value=0, max_value=999, value=999)
        with c:
            previous = st.number_input("Contacts Before This Campaign", min_value=0, max_value=100, value=0)
            poutcome = st.selectbox("Previous Campaign Outcome", sorted(df["poutcome"].dropna().unique()))
            st.caption("999 for previous contact days means the customer was not previously contacted.")

        st.subheader("Economic Indicators")
        a, b, c = st.columns(3)
        with a:
            emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, format="%.3f")
            cons_price_idx = st.number_input("Consumer Price Index", value=93.994, format="%.3f")
        with b:
            cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, format="%.3f")
            euribor3m = st.number_input("Euribor 3-Month Rate", value=4.857, format="%.3f")
        with c:
            nr_employed = st.number_input("Number of Employees", value=5191.0, format="%.3f")

        submitted = st.form_submit_button("🔮 Predict Subscription", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day_of_week, "campaign": campaign,
            "pdays": pdays, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "nr.employed": nr_employed
        }], columns=features)
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0, 1])
        if prediction == 1:
            st.markdown(f"""<div class="result-yes"><h2>Likely to Subscribe (YES)</h2>
            <p>The model classifies this customer as likely to subscribe.</p></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-no"><h2>Unlikely to Subscribe (NO)</h2>
            <p>The model classifies this customer as unlikely to subscribe.</p></div>""",
                        unsafe_allow_html=True)
        x, y = st.columns(2)
        x.metric("Estimated Subscription Probability", f"{probability:.1%}")
        y.metric("Estimated Non-subscription Probability", f"{1-probability:.1%}")
        st.progress(probability)
        st.caption("This is a machine-learning estimate, not a guarantee or a financial recommendation.")

elif page == "About Project":
    st.title("About This Project")
    st.markdown("""
**Project:** Bank Customer Term-Deposit Subscription Prediction

**Objective:** Predict whether a bank customer is likely to subscribe to a term deposit after a marketing campaign.

**Dataset:** UCI Bank Marketing (`bank-additional-full.csv`).

**Machine-learning task:** Binary classification (`yes` / `no`).

**Models compared:** Logistic Regression, Decision Tree and Random Forest.

**Preprocessing:** Median imputation and scaling for numeric columns; most-frequent imputation and one-hot encoding for categorical columns.

**Evaluation:** Accuracy, Precision, Recall, F1-score and ROC-AUC.

**Important:** `duration` is intentionally excluded because it is known only after the call and may cause data leakage.
""")
    st.write(f"**Selected model:** {bundle['model_name']}")
    st.write(f"**Training rows:** {bundle['train_size']:,} | **Testing rows:** {bundle['test_size']:,}")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

st.divider()
st.caption("Academic project demonstration • Predictions are estimates based on the UCI dataset.")
