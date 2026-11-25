import io
import zipfile
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Bank Customer Prediction", page_icon="🏦", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
MODEL_FILE = Path("bank_term_deposit_model.pkl")

@st.cache_data(show_spinner="Loading the bank marketing dataset...")
def load_data():
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        nested_name = next((n for n in outer.namelist() if n.endswith("bank-additional.zip")), None)
        if nested_name:
            nested_bytes = outer.read(nested_name)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                csv_name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
                with inner.open(csv_name) as csv_file:
                    data = pd.read_csv(csv_file, sep=";")
        else:
            csv_name = next(n for n in outer.namelist() if n.endswith("bank-additional-full.csv"))
            with outer.open(csv_name) as csv_file:
                data = pd.read_csv(csv_file, sep=";")
    return data

@st.cache_resource(show_spinner="Training and preparing the machine-learning models (first run may take a few minutes)...")
def train_models(data):
    X = data.drop(columns=["y", "duration"]).copy()  # duration is excluded to avoid data leakage
    y = (data["y"] == "yes").astype(int)
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
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    trained, rows = {}, []
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        trained[name] = pipe
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, prob)
        })
    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best_model = trained[best_name]
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]
    return X, trained, results, best_name, best_model, y_test, best_pred, best_prob


def main():
    st.markdown("# 🏦 Bank Customer Term-Deposit Prediction")
    st.caption("Applied Predictive Analysis | Group P1 E | Machine-learning based customer prediction")
    st.info("Enter a customer's details to estimate whether they may subscribe to a bank term deposit. This is an academic model estimate, not a guarantee or a real banking decision.")

    try:
        df = load_data()
        X, trained, results, best_name, best_model, y_test, best_pred, best_prob = train_models(df)
    except Exception as exc:
        st.error("The dataset or model could not be loaded. Check your internet connection and restart the app.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

    total = len(df)
    yes_count = int((df["y"] == "yes").sum())
    no_count = int((df["y"] == "no").sum())
    rate = yes_count / total * 100

    tabs = st.tabs(["Customer Prediction", "Prediction History", "Analytics Dashboard", "Model Performance", "About Project"])

    # Session history: entries remain available while this browser session is active.
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    with tabs[0]:
        st.subheader("Customer details")
        st.write("Fill in the fields below and click **Predict Subscription**.")
        numeric_defaults = {
            "age": 35, "campaign": 1, "pdays": 999, "previous": 0,
            "emp.var.rate": 1.1, "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
        }
        with st.form("customer_form"):
            values = {}
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
            left, right = st.columns(2)
            for i, col in enumerate(numeric_cols):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                default = float(numeric_defaults.get(col, vals.median()))
                with (left if i % 2 == 0 else right):
                    if col == "age":
                        values[col] = st.number_input("Age (years)", min_value=17, max_value=100, value=int(default), step=1)
                    elif col in ("campaign", "pdays", "previous"):
                        values[col] = st.number_input(col.replace(".", " ").title(), min_value=0, max_value=9999, value=int(default), step=1)
                    else:
                        values[col] = st.number_input(col.replace(".", " ").title(), value=default, format="%.3f")
            st.markdown("#### Customer and campaign information")
            cat_cols = st.columns(2)
            for i, col in enumerate(categorical_cols):
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                mode_values = df[col].mode()
                default_value = str(mode_values.iloc[0]) if not mode_values.empty else options[0]
                with cat_cols[i % 2]:
                    values[col] = st.selectbox(col.replace(".", " ").title(), options,
                                               index=options.index(default_value) if default_value in options else 0)
            submitted = st.form_submit_button("🔍 Predict Subscription", type="primary", use_container_width=True)

        if submitted:
            customer = pd.DataFrame([{col: values[col] for col in X.columns}], columns=X.columns)
            try:
                prediction = int(best_model.predict(customer)[0])
                probability = float(best_model.predict_proba(customer)[0, 1])
                st.session_state["prediction_history"].append({
                    "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": values.get("age"),
                    "Job": values.get("job"),
                    "Marital": values.get("marital"),
                    "Education": values.get("education"),
                    "Campaign": values.get("campaign"),
                    "Prediction": "YES" if prediction == 1 else "NO",
                    "Subscription Probability (%)": round(probability * 100, 2),
                    "Model": best_name
                })
                st.divider()
                st.subheader("Prediction result")
                a, b = st.columns(2)
                with a:
                    if prediction == 1:
                        st.success("YES — likely to subscribe")
                    else:
                        st.warning("NO — unlikely to subscribe")
                with b:
                    st.metric("Estimated subscription probability", f"{probability:.2%}")
                st.progress(probability)
                st.caption(f"Prediction generated using the selected {best_name} model. Probability is an estimate, not a guarantee.")
            except Exception as exc:
                st.error("Prediction could not be generated. Please check the entered values and try again.")
                st.code(str(exc))

    with tabs[1]:
        st.subheader("Customer prediction history")
        st.caption("Predictions made during this active browser session are listed here.")

        history = st.session_state["prediction_history"]
        if history:
            history_df = pd.DataFrame(history)
            total_checks = len(history_df)
            yes_checks = int((history_df["Prediction"] == "YES").sum())
            no_checks = total_checks - yes_checks

            h1, h2, h3 = st.columns(3)
            h1.metric("Total predictions", total_checks)
            h2.metric("Predicted YES", yes_checks)
            h3.metric("Predicted NO", no_checks)

            st.dataframe(history_df, use_container_width=True, hide_index=True)
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download prediction history (CSV)",
                data=csv_data,
                file_name="bank_customer_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )
            if st.button("Clear prediction history", type="secondary"):
                st.session_state["prediction_history"] = []
                st.rerun()
        else:
            st.info("No predictions recorded yet. Go to Customer Prediction, enter customer details and click Predict Subscription.")

    with tabs[2]:
        st.subheader("Bank marketing overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total customers", f"{total:,}")
        c2.metric("Subscribed", f"{yes_count:,}")
        c3.metric("Not subscribed", f"{no_count:,}")
        c4.metric("Subscription rate", f"{rate:.2f}%")
        chart1, chart2 = st.columns(2)
        with chart1:
            counts = df["y"].value_counts().rename_axis("Subscription").reset_index(name="Customers")
            st.plotly_chart(px.pie(counts, names="Subscription", values="Customers", hole=0.45, title="Subscription distribution"), use_container_width=True)
        with chart2:
            job_summary = df.assign(subscribed=(df["y"] == "yes").astype(int)).groupby("job", as_index=False)["subscribed"].mean()
            job_summary["Subscription rate (%)"] = job_summary["subscribed"] * 100
            st.plotly_chart(px.bar(job_summary.sort_values("Subscription rate (%)", ascending=False), x="job", y="Subscription rate (%)", title="Subscription rate by job"), use_container_width=True)
        month_summary = df.assign(subscribed=(df["y"] == "yes").astype(int)).groupby("month", as_index=False)["subscribed"].mean()
        month_summary["Subscription rate (%)"] = month_summary["subscribed"] * 100
        month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month_summary["month"] = pd.Categorical(month_summary["month"], categories=month_order, ordered=True)
        month_summary = month_summary.sort_values("month")
        st.plotly_chart(px.line(month_summary, x="month", y="Subscription rate (%)", markers=True, title="Subscription rate by month"), use_container_width=True)
        job_filter = st.selectbox("Explore customers by job", ["All"] + sorted(df["job"].dropna().unique().tolist()))
        filtered = df if job_filter == "All" else df[df["job"] == job_filter]
        st.caption(f"Customers in selection: {len(filtered):,} | Subscribed: {(filtered['y'] == 'yes').sum():,} | Subscription rate: {(filtered['y'] == 'yes').mean():.2%}")
        st.dataframe(filtered[["age", "job", "marital", "education", "housing", "loan", "y"]].head(15), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Model comparison")
        st.dataframe(results.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]}), use_container_width=True, hide_index=True)
        long = results.melt(id_vars="Model", var_name="Metric", value_name="Score", value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], title="Performance of the three algorithms"), use_container_width=True)
        cm = confusion_matrix(y_test, best_pred)
        cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])
        st.write(f"**Selected model:** {best_name}")
        st.write("Confusion matrix")
        st.dataframe(cm_df, use_container_width=True)
        st.caption(f"Test ROC-AUC: {roc_auc_score(y_test, best_prob):.3f}")

    with tabs[4]:
        st.subheader("Project information")
        st.markdown("""
        **Objective:** Predict whether a bank customer is likely to subscribe to a term deposit based on customer, contact and campaign information.

        **Dataset:** UCI Bank Marketing (`bank-additional-full.csv`), containing 41,188 records and 20 input features plus the target.

        **Algorithms:** Logistic Regression, Decision Tree and Random Forest. The app selects the model with the highest F1-score on the held-out test set.

        **Important modelling decision:** The `duration` field is deliberately excluded because call duration is only known after a call. Including it would cause data leakage for a prediction intended to be made before contacting a customer.

        **Limitations:** The dataset is historical and the target classes are imbalanced. Predictions should be treated as educational estimates, not financial advice or real customer decisions.

        **Reference:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository.
        """)

    st.sidebar.markdown("### Project details")
    st.sidebar.write("**Group:** P1 E")
    st.sidebar.write("**Platform:** Python / Streamlit")
    st.sidebar.write("**Best model (current run):** " + best_name)
    st.sidebar.caption("First load trains the models. Later interactions use the cached model.")
    st.sidebar.caption("Prediction history is kept for the active browser session and can be downloaded as CSV.")

if __name__ == "__main__":
    main()
