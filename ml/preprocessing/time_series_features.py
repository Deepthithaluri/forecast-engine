from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_daily_sales() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DATA_PATH / "daily_sales.csv",
        parse_dates=["date"],
    )


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    df["lag_1"] = df["total_sales"].shift(1)
    df["lag_7"] = df["total_sales"].shift(7)

    df["rolling_mean_7"] = (
        df["total_sales"]
        .rolling(window=7)
        .mean()
    )

    df["rolling_std_7"] = (
        df["total_sales"]
        .rolling(window=7)
        .std()
    )

    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.dayofweek

    df = df.dropna()

    return df


def save_dataset(df: pd.DataFrame):
    df.to_csv(
        PROCESSED_DATA_PATH / "daily_sales_features.csv",
        index=False,
    )


def main():
    df = load_daily_sales()

    featured_df = create_time_series_features(df)

    save_dataset(featured_df)

    print("Time-series feature engineering completed successfully.")
    print(featured_df.head())


if __name__ == "__main__":
    main()