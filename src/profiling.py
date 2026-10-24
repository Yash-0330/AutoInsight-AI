
import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the uploaded or loaded dataset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.shape[1] > 500:
        raise ValueError("Dataset cannot contain more than 500 columns.")

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate a basic data quality and statistical profile."""
    validate_dataframe(df)

    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    # Count duplicate rows, including rows containing missing values.
    duplicate_rows = int(df.duplicated(keep="first").sum())

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Count": df.isna().sum().values,
        "Missing (%)": (df.isna().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "column_quality": quality,
        "numeric_summary": numeric.describe().T if not numeric.empty else pd.DataFrame(),
        "categorical_summary": (
            categorical.describe().T
            if not categorical.empty
            else pd.DataFrame()
        ),
    }


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values and impute missing numeric values."""
    numeric = df.select_dtypes(include="number").copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    for column in numeric.columns:
        median = numeric[column].median()
        numeric[column] = numeric[column].fillna(
            median if pd.notna(median) else 0
        )

    return numeric
