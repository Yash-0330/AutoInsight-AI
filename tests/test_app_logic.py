
import pandas as pd
import pytest

from src.profiling import profile_dataframe, validate_dataframe
from src.anomaly_detection import detect_anomalies
from src.modeling import compare_models
from src.reporting import build_markdown_report


def test_profile_counts():
    df = pd.DataFrame({
        "amount": [10, 20, 20],
        "region": ["West", None, "West"],
    })

    result = profile_dataframe(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 0


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())


def test_anomaly_detection():
    df = pd.DataFrame({
        "amount": list(range(30)) + [1000],
        "units": list(range(30)) + [1000],
    })

    result = detect_anomalies(
        df,
        columns=["amount", "units"],
        contamination=0.05,
    )

    assert "Anomaly Status" in result.columns
    assert "Anomaly Score" in result.columns
    assert len(result) == len(df)


def test_classification_model_comparison():
    df = pd.DataFrame({
        "feature_a": [i % 10 for i in range(100)],
        "feature_b": [i / 3 for i in range(100)],
        "target": ["Yes" if i % 2 else "No" for i in range(100)],
    })

    scores, task, predictions = compare_models(df, "target")

    assert task == "classification"
    assert not scores.empty
    assert "F1 (Macro)" in scores.columns
    assert {"Actual", "Predicted"}.issubset(predictions.columns)


def test_report_generation():
    df = pd.DataFrame({"amount": [10, 20, 30]})
    profile = profile_dataframe(df)

    report = build_markdown_report(df, profile)

    assert "Dataset Overview" in report
    assert "Rows: 3" in report
