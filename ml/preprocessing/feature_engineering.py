from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_cleaned_orders():
    return pd.read_csv(
        PROCESSED_DATA_PATH / "cleaned_orders.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.day_name()

    df["delivery_time_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.days

    return df


def save_featured_data(df: pd.DataFrame):
    df.to_csv(
        PROCESSED_DATA_PATH / "featured_orders.csv",
        index=False,
    )


def main():
    orders_df = load_cleaned_orders()
    featured_df = create_features(orders_df)
    save_featured_data(featured_df)

    print("Feature engineering completed successfully.")
    print(featured_df.head())


if __name__ == "__main__":
    main()