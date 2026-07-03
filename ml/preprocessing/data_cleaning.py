from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_orders():
    return pd.read_csv(RAW_DATA_PATH / "olist_orders_dataset.csv")


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df = df.drop_duplicates()

    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH / filename, index=False)


def main():
    orders_df = load_orders()
    cleaned_orders_df = clean_orders(orders_df)
    save_processed_data(cleaned_orders_df, "cleaned_orders.csv")

    print("Cleaning completed successfully.")
    print(f"Rows: {cleaned_orders_df.shape[0]}")
    print(f"Columns: {cleaned_orders_df.shape[1]}")


if __name__ == "__main__":
    main()