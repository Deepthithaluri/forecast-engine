from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

TRAIN_FILE = Path("ml/data/processed/train.csv")
TEST_FILE = Path("ml/data/processed/test.csv")
MODEL_FILE = Path("ml/models/forecasting_model.joblib")

METRICS_DIR = Path("ml/metrics")
PREDICTIONS_DIR = Path("ml/predictions")

METRICS_FILE = METRICS_DIR / "forecasting_metrics.json"
PREDICTIONS_FILE = PREDICTIONS_DIR / "forecasting_predictions.csv"

TARGET_COLUMN = "daily_sales"

DROP_COLUMNS = [
    "Order Date",
    TARGET_COLUMN,
]


def load_dataset(file_path: Path, dataset_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    df = pd.read_csv(file_path, parse_dates=["Order Date"])

    logger.info("Loaded %d %s rows", len(df), dataset_name)

    return df


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df["Order Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    if not df["Order Date"].is_monotonic_increasing:
        raise ValueError("Dataset must be sorted by date.")


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"{MODEL_FILE} not found.")

    logger.info("Loading trained model")

    return joblib.load(MODEL_FILE)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN]

    return X, y


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    non_zero_mask = y_true != 0

    if non_zero_mask.any():
        mape = (
            mean_absolute_percentage_error(
                y_true[non_zero_mask],
                y_pred[non_zero_mask],
            )
            * 100
        )
    else:
        mape = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": None if mape is None else float(mape),
        "r2": float(r2),
    }


def save_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df = pd.DataFrame(
        {
            "Order Date": df["Order Date"],
            "Actual": df[TARGET_COLUMN],
            "Predicted": predictions,
        }
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    logger.info(
        "Saved predictions to %s",
        PREDICTIONS_FILE,
    )


def save_metrics(
    metrics: dict,
    train_rows: int,
    test_rows: int,
    feature_count: int,
) -> None:

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_output = {
        "model": "XGBoost Regressor",
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": feature_count,
        **metrics,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_output,
            file,
            indent=4,
        )

    logger.info(
        "Saved metrics to %s",
        METRICS_FILE,
    )


def print_metrics(metrics: dict) -> None:

    print("\nForecasting Model Evaluation\n")

    print(f"MAE   : ₹{metrics['mae']:,.2f}")
    print(f"RMSE  : ₹{metrics['rmse']:,.2f}")

    if metrics["mape"] is None:
        print("MAPE  : N/A")
    else:
        print(f"MAPE  : {metrics['mape']:.2f}%")

    print(f"R²    : {metrics['r2']:.4f}")


def main() -> None:

    train_df = load_dataset(TRAIN_FILE, "training")

    test_df = load_dataset(TEST_FILE, "testing")

    validate_data(train_df)
    validate_data(test_df)

    model = load_model()

    X_test, y_test = prepare_features(test_df)

    predictions = model.predict(X_test)

    metrics = evaluate(
        y_test,
        predictions,
    )

    save_predictions(
        test_df,
        predictions,
    )

    save_metrics(
        metrics=metrics,
        train_rows=len(train_df),
        test_rows=len(test_df),
        feature_count=X_test.shape[1],
    )

    print_metrics(metrics)


if __name__ == "__main__":
    main()