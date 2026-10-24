
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def infer_task_type(y):
    if (
        not pd.api.types.is_numeric_dtype(y)
        or y.nunique(dropna=True) <= 10
    ):
        return "classification"

    return "regression"


def compare_models(df, target, test_size=0.2):
    if target not in df.columns:
        raise ValueError("Target column not found.")

    work = df.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[target]).copy()

    y = work[target]
    X = work.drop(columns=[target])

    X = X.loc[:, X.nunique(dropna=True) > 1]

    if len(work) < 30:
        raise ValueError("At least 30 usable rows are required.")

    if y.nunique() < 2:
        raise ValueError("Target must have at least two values.")

    if X.empty:
        raise ValueError("No usable predictor columns remain.")

    task = infer_task_type(y)

    if task == "classification":
        counts = y.value_counts()

        if len(counts) > 50:
            raise ValueError("Too many target classes.")

        if counts.min() < 2:
            raise ValueError(
                "Each class needs at least two records."
            )

        stratify = y
    else:
        stratify = None

    # Exclude very high-cardinality text columns for this baseline.
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    high_cardinality = [
        col for col in categorical_cols
        if X[col].nunique(dropna=True)
        > min(100, max(20, int(len(X) * 0.5)))
    ]

    X = X.drop(columns=high_cardinality)

    if X.empty:
        raise ValueError("No usable predictors remain.")

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=50,
                )),
            ]),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": HistGradientBoostingClassifier(
                random_state=42
            ),
        }
    else:
        models = {
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": HistGradientBoostingRegressor(
                random_state=42
            ),
        }

    results = []
    predictions_by_model = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            predictions_by_model[name] = predictions

            if task == "classification":
                row = {
                    "Model": name,
                    "Accuracy": accuracy_score(
                        y_test, predictions
                    ),
                    "Precision (Macro)": precision_score(
                        y_test, predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "Recall (Macro)": recall_score(
                        y_test, predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "F1 (Macro)": f1_score(
                        y_test, predictions,
                        average="macro",
                        zero_division=0,
                    ),
                }

                if y.nunique() == 2:
                    try:
                        probabilities = pipeline.predict_proba(
                            X_test
                        )[:, 1]

                        positive_label = (
                            pipeline.named_steps["model"].classes_[1]
                        )

                        binary_truth = (
                            y_test == positive_label
                        ).astype(int)

                        row["ROC-AUC"] = roc_auc_score(
                            binary_truth, probabilities
                        )
                    except (ValueError, AttributeError, IndexError):
                        pass

            else:
                row = {
                    "Model": name,
                    "MAE": mean_absolute_error(
                        y_test, predictions
                    ),
                    "RMSE": float(np.sqrt(
                        mean_squared_error(y_test, predictions)
                    )),
                    "R2": r2_score(y_test, predictions),
                }

            results.append(row)

        except Exception as exc:
            results.append({
                "Model": name,
                "Error": str(exc),
            })

    successful = [
        row for row in results if "Error" not in row
    ]

    if not successful:
        raise ValueError("All candidate models failed.")

    scores = pd.DataFrame(successful)

    if task == "classification":
        scores = scores.sort_values(
            "F1 (Macro)", ascending=False
        )
    else:
        scores = scores.sort_values("RMSE", ascending=True)

    scores = scores.reset_index(drop=True)
    best_name = scores.iloc[0]["Model"]

    prediction_output = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions_by_model[best_name],
    })

    return scores, task, prediction_output
