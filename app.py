
import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.profiling import profile_dataframe
from src.anomaly_detection import detect_anomalies
from src.modeling import compare_models
from src.reporting import (
    build_markdown_report,
    optional_llm_explanation,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "data" / "sample_data.csv"
DB_PATH = BASE_DIR / "autoinsight.db"

st.set_page_config(
    page_title="AutoInsight AI",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AutoInsight AI")
st.caption(
    "Autonomous Data Science Agent | "
    "Profile • Explore • Detect • Model • Explain"
)

with st.sidebar:
    st.header("Dataset")

    source = st.radio(
        "Choose data source",
        ["Sample dataset", "Upload CSV"],
    )

    uploaded_file = None

    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
        )

try:
    if source == "Sample dataset":
        df = pd.read_csv(SAMPLE_PATH)

    elif uploaded_file is not None:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("Maximum file size is 50 MB.")
            st.stop()

        df = pd.read_csv(uploaded_file, low_memory=False)

    else:
        st.info("Upload a CSV or select the sample dataset.")
        st.stop()

except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

df = df.dropna(how="all").dropna(axis=1, how="all")

try:
    profile = profile_dataframe(df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if len(df.columns) > 200:
    st.error("This application supports up to 200 columns.")
    st.stop()

overview, explore, anomalies, machine_learning, sql, report = st.tabs([
    "Overview",
    "Explore Data",
    "Anomaly Detection",
    "Machine Learning",
    "SQL Explorer",
    "AI Report",
])

# 1. Overview
with overview:
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", f"{profile['columns']:,}")
    c3.metric("Missing Cells", f"{profile['missing_cells']:,}")
    c4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")

    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Column Quality")
    st.dataframe(profile["column_quality"], use_container_width=True)

    st.download_button(
        "Download Data Quality Report",
        profile["column_quality"].to_csv(index=False).encode("utf-8"),
        file_name="data_quality_report.csv",
        mime="text/csv",
    )

# 2. EDA
with explore:
    st.header("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    left, right = st.columns(2)

    with left:
        st.subheader("Numeric Statistics")
        st.dataframe(
            profile["numeric_summary"],
            use_container_width=True,
        )

    with right:
        st.subheader("Categorical Statistics")
        st.dataframe(
            profile["categorical_summary"],
            use_container_width=True,
        )

    if numeric_cols:
        selected_num = st.selectbox(
            "Select a numeric column",
            numeric_cols,
        )

        values = pd.to_numeric(
            df[selected_num], errors="coerce"
        ).dropna()

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(values, kde=len(values) > 2, ax=ax)
        ax.set_title(f"Distribution of {selected_num}")
        st.pyplot(fig)
        plt.close(fig)

    if categorical_cols:
        selected_cat = st.selectbox(
            "Select a categorical column",
            categorical_cols,
        )

        counts = (
            df[selected_cat]
            .astype("string")
            .fillna("(missing)")
            .value_counts()
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Top values: {selected_cat}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")

        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

# 3. Anomaly Detection
with anomalies:
    st.header("Anomaly Detection")

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_cols) > 0 and len(df) >= 10:
        selected_cols = st.multiselect(
            "Choose numeric features",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
        )

        contamination = st.slider(
            "Expected anomaly proportion",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
        )

        if st.button("Detect Anomalies"):
            try:
                results = detect_anomalies(
                    df,
                    columns=selected_cols,
                    contamination=contamination,
                )

                flagged = results[
                    results["Anomaly Status"] == "Potential Anomaly"
                ]

                st.metric("Potential Anomalies", len(flagged))
                st.dataframe(flagged, use_container_width=True)

                st.download_button(
                    "Download Anomaly Results",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="anomaly_results.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(str(exc))

    else:
        st.info(
            "At least 10 rows and one numeric column are required."
        )

    st.warning(
        "Anomaly detection flags unusual records; it does not "
        "confirm fraud, mistakes, or malicious activity."
    )

# 4. Machine Learning
with machine_learning:
    st.header("Automatic Model Comparison")

    st.write(
        "Choose a target column that you want to predict. "
        "The application will compare baseline models."
    )

    target = st.selectbox(
        "Target column",
        ["-- Select target --"] + df.columns.tolist(),
    )

    if st.button(
        "Train and Compare Models",
        disabled=(target == "-- Select target --"),
    ):
        try:
            with st.spinner("Training models..."):
                scores, task, predictions = compare_models(
                    df, target
                )

            st.success(f"Detected task type: {task}")

            st.subheader("Model Evaluation")
            st.dataframe(scores, use_container_width=True)

            st.subheader("Best Model Predictions")
            st.dataframe(
                predictions.head(100),
                use_container_width=True,
            )

            st.download_button(
                "Download Model Scores",
                scores.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
            )

            st.download_button(
                "Download Predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="test_predictions.csv",
                mime="text/csv",
            )

            st.session_state["model_scores"] = scores
            st.session_state["model_task"] = task

        except Exception as exc:
            st.error(f"Model training failed: {exc}")

# 5. SQL Explorer
with sql:
    st.header("SQLite Data Explorer")

    st.write(
        "Save the current dataset to SQLite and run a single "
        "read-only SELECT query."
    )

    if st.button("Save Dataset to SQLite"):
        try:
            with sqlite3.connect(DB_PATH) as connection:
                df.to_sql(
                    "uploaded_data",
                    connection,
                    if_exists="replace",
                    index=False,
                )

            st.success("Dataset saved to the uploaded_data table.")

        except Exception as exc:
            st.error(f"SQLite save failed: {exc}")

    query = st.text_area(
        "SQL Query",
        "SELECT * FROM uploaded_data LIMIT 10",
    )

    if st.button("Run SQL Query"):
        normalized = query.strip().lower()

        if (
            not normalized.startswith("select")
            or ";" in normalized.rstrip(";")
        ):
            st.error("Only one SELECT statement is allowed.")

        elif not DB_PATH.exists():
            st.error("Save the dataset to SQLite first.")

        else:
            try:
                uri = f"file:{DB_PATH.as_posix()}?mode=ro"

                with sqlite3.connect(uri, uri=True) as connection:
                    result = pd.read_sql_query(query, connection)

                st.dataframe(result, use_container_width=True)

            except Exception as exc:
                st.error(f"SQL query failed: {exc}")

# 6. Reporting and optional LLM
with report:
    st.header("Analysis Report")

    scores = st.session_state.get("model_scores")
    task = st.session_state.get("model_task")

    markdown_report = build_markdown_report(
        df,
        profile,
        scores=scores,
        task_type=task,
    )

    st.markdown(markdown_report)

    st.download_button(
        "Download Markdown Report",
        markdown_report.encode("utf-8"),
        file_name="autoinsight_report.md",
        mime="text/markdown",
    )

    st.divider()
    st.subheader("Optional AI Explanation")

    st.write(
        "Configure your LLM API credentials in .env to enable "
        "natural-language explanations. Core analysis works without them."
    )

    if st.button("Generate AI Explanation"):
        summary = (
            f"Dataset rows: {profile['rows']}; "
            f"columns: {profile['columns']}; "
            f"missing cells: {profile['missing_cells']}; "
            f"duplicate rows: {profile['duplicate_rows']}. "
            f"Column summary: "
            f"{profile['quality'].head(50).to_dict(orient='records')}. "
        )

        if scores is not None:
            summary += f"Model scores: {scores.to_dict(orient='records')}."

        with st.spinner("Generating explanation..."):
            explanation, error = optional_llm_explanation(summary)

        if error:
            st.info(error)
        else:
            st.markdown(explanation)

            st.download_button(
                "Download AI Explanation",
                explanation.encode("utf-8"),
                file_name="ai_explanation.md",
                mime="text/markdown",
            )

st.divider()
st.caption(
    "AutoInsight AI — Educational prototype. Validate all findings "
    "before making important decisions."
)
