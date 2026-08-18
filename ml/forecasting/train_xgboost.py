from pathlib import Path
import logging

import joblib
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

TRAIN_FILE = Path("ml/data/processed/train.csv")

MODEL_DIR = Path("ml/models")
MODEL_FILE = MODEL_DIR / "forecasting_model.joblib"

TARGET_COLUMN = "daily_sales"

DROP_COLUMNS = [
    "Order Date",
    TARGET_COLUMN,
]

MODEL_CONFIG = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def load_data() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found.")

    df = pd.read_csv(TRAIN_FILE, parse_dates=["Order Date"])

    logger.info("Loaded %d training rows", len(df))

    return df


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Training dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN]

    return X, y


def build_model() -> XGBRegressor:
    return XGBRegressor(**MODEL_CONFIG)


def train_model(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> XGBRegressor:
    logger.info("Training XGBoost model...")

    model.fit(X, y)

    return model


def save_model(model: XGBRegressor) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    logger.info("Saved model to %s", MODEL_FILE)


def main() -> None:
    df = load_data()

    validate_data(df)

    X, y = prepare_features(df)

    model = build_model()

    model = train_model(model, X, y)

    save_model(model)

    print("\nTraining Summary")
    print(f"Training Rows : {len(df)}")
    print(f"Features      : {X.shape[1]}")
    print(f"Model Saved   : {MODEL_FILE}")


if __name__ == "__main__":
    main()