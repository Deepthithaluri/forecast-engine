from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_integrated_dataset() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DATA_PATH / "integrated_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    daily_sales = (
        df.groupby(df["order_purchase_timestamp"].dt.date)
        .agg(
            total_sales=("payment_value", "sum"),
            total_orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    daily_sales.rename(
        columns={"order_purchase_timestamp": "date"},
        inplace=True,
    )

    return daily_sales


def save_dataset(df: pd.DataFrame):
    df.to_csv(
        PROCESSED_DATA_PATH / "daily_sales.csv",
        index=False,
    )


def main():
    df = load_integrated_dataset()

    daily_sales = aggregate_daily_sales(df)

    save_dataset(daily_sales)

    print("Daily sales dataset created successfully.")
    print(daily_sales.head())


if __name__ == "__main__":
    main()