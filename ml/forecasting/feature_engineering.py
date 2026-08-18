from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

INPUT_FILE = Path("ml/data/processed/daily_sales.csv")
OUTPUT_FILE = Path("ml/data/processed/training_dataset.csv")

REQUIRED_COLUMNS = {
    "Order Date",
    "daily_sales",
    "total_orders",
    "total_quantity",
    "total_profit",
}


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} does not exist.")

    df = pd.read_csv(INPUT_FILE, parse_dates=["Order Date"])
    logger.info("Loaded %d records", len(df))
    return df


def validate_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Order Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    if df["daily_sales"].isna().any():
        raise ValueError("daily_sales contains missing values.")


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["daily_sales"].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in [7, 14, 28]:
        shifted = df["daily_sales"].shift(1)

        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()

    return df


def create_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df["daily_sales"].shift(1)

    df["expanding_mean"] = shifted.expanding().mean()
    df["expanding_std"] = shifted.expanding().std()

    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("No data remaining after feature engineering.")

    if df.isna().sum().sum() > 0:
        raise ValueError("Dataset still contains missing values.")

    return df


def save_dataset(df: pd.DataFrame) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Saved training dataset to %s", OUTPUT_FILE)


def print_summary(df: pd.DataFrame) -> None:
    print("\nTraining Dataset Summary")
    print(f"Rows              : {len(df)}")
    print(f"Columns           : {len(df.columns)}")
    print(f"Start Date        : {df['Order Date'].min().date()}")
    print(f"End Date          : {df['Order Date'].max().date()}")
    print(f"Missing Values    : {df.isna().sum().sum()}")
    print(f"Feature Columns   : {len(df.columns) - 1}")


def main() -> None:
    df = load_data()

    validate_data(df)

    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_expanding_features(df)
    df = create_cyclical_features(df)

    df = finalize_dataset(df)

    save_dataset(df)

    print_summary(df)


if __name__ == "__main__":
    main()