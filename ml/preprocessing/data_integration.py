from pathlib import Path

import pandas as pd

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a dataset from the raw data directory."""
    return pd.read_csv(RAW_DATA_PATH / filename)


def integrate_data() -> pd.DataFrame:
    """Merge all required datasets into a single integrated dataset."""

    # Load datasets
    orders = load_dataset("olist_orders_dataset.csv")
    customers = load_dataset("olist_customers_dataset.csv")
    order_items = load_dataset("olist_order_items_dataset.csv")
    products = load_dataset("olist_products_dataset.csv")
    payments = load_dataset("olist_order_payments_dataset.csv")
    translation = load_dataset(
        "product_category_name_translation.csv"
    )

    # Orders + Customers
    merged_df = orders.merge(
        customers,
        on="customer_id",
        how="left",
    )

    # + Order Items
    merged_df = merged_df.merge(
        order_items,
        on="order_id",
        how="left",
    )

    # + Products
    merged_df = merged_df.merge(
        products,
        on="product_id",
        how="left",
    )

    # + Category Translation
    merged_df = merged_df.merge(
        translation,
        on="product_category_name",
        how="left",
    )

    # + Payments
    merged_df = merged_df.merge(
        payments,
        on="order_id",
        how="left",
    )

    return merged_df


def save_data(df: pd.DataFrame):
    """Save integrated dataset."""

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA_PATH / "integrated_dataset.csv",
        index=False,
    )


def main():
    integrated_df = integrate_data()

    save_data(integrated_df)

    print("=" * 60)
    print("Dataset integration completed successfully!")
    print("=" * 60)
    print(f"Rows    : {integrated_df.shape[0]}")
    print(f"Columns : {integrated_df.shape[1]}")

    print("\nColumns:")
    for column in integrated_df.columns:
        print(f"- {column}")

    print("\nFirst Five Rows:")
    print(integrated_df.head())


if __name__ == "__main__":
    main()