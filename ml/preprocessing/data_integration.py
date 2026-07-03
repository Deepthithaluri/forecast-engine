from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH / filename)


def integrate_data() -> pd.DataFrame:
    orders = load_dataset("olist_orders_dataset.csv")
    customers = load_dataset("olist_customers_dataset.csv")
    order_items = load_dataset("olist_order_items_dataset.csv")
    products = load_dataset("olist_products_dataset.csv")
    payments = load_dataset("olist_order_payments_dataset.csv")

    merged_df = orders.merge(
        customers,
        on="customer_id",
        how="left",
    )

    merged_df = merged_df.merge(
        order_items,
        on="order_id",
        how="left",
    )

    merged_df = merged_df.merge(
        products,
        on="product_id",
        how="left",
    )

    merged_df = merged_df.merge(
        payments,
        on="order_id",
        how="left",
    )

    return merged_df


def save_data(df: pd.DataFrame):
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA_PATH / "integrated_dataset.csv",
        index=False,
    )


def main():
    integrated_df = integrate_data()

    save_data(integrated_df)

    print("Dataset integration completed successfully.")
    print(f"Rows    : {integrated_df.shape[0]}")
    print(f"Columns : {integrated_df.shape[1]}")

    print("\nFirst Five Rows")
    print(integrated_df.head())


if __name__ == "__main__":
    main()