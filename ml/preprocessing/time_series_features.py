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
    """
    Create lag and rolling features for XGBoost forecasting.
    """

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

    df = df.dropna().reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA_PATH / "daily_sales_features.csv",
        index=False,
    )


def main():

    df = load_daily_sales()

    featured_df = create_time_series_features(df)

    save_dataset(featured_df)

    print("=" * 60)
    print("Time-Series Feature Engineering Completed")
    print("=" * 60)
    print(featured_df.head())

    print("\nRows :", len(featured_df))
    print("Columns :", len(featured_df.columns))


if __name__ == "__main__":
    main()