
import pandas as pd
from sklearn.ensemble import IsolationForest
from .profiling import clean_numeric_frame


def detect_anomalies(
    df,
    columns=None,
    contamination=0.05,
    random_state=42,
):
    if len(df) < 10:
        raise ValueError("At least 10 rows are required.")

    if not 0.01 <= contamination <= 0.20:
        raise ValueError("Contamination must be between 0.01 and 0.20.")

    if columns is not None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        selected_df = df[columns]
    else:
        selected_df = df

    features = clean_numeric_frame(selected_df)

    if features.empty:
        raise ValueError("Select at least one numeric column.")

    if features.shape[1] == 0:
        raise ValueError("No numeric features are available.")

    if features.isna().all().any():
        raise ValueError("Selected columns contain no usable numeric values.")

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )

    predictions = model.fit_predict(features)
    scores = model.decision_function(features)

    result = df.copy()
    result["Anomaly Status"] = [
        "Potential Anomaly" if value == -1 else "Normal"
        for value in predictions
    ]
    result["Anomaly Score"] = scores

    return result
