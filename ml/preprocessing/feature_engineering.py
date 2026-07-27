from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_cleaned_orders() -> pd.DataFrame:
    """
    Load cleaned UCI Online Retail dataset.
    """
    return pd.read_csv(
        PROCESSED_DATA_PATH / "cleaned_orders.csv",
        parse_dates=["InvoiceDate"],
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create useful features for analysis and recommendation.
    """

    df["Sales"] = df["Quantity"] * df["Price"]

    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["Quarter"] = df["InvoiceDate"].dt.quarter

    return df


def save_featured_data(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA_PATH / "featured_orders.csv",
        index=False,
    )


def main():

    orders_df = load_cleaned_orders()

    featured_df = create_features(orders_df)

    save_featured_data(featured_df)

    print("=" * 60)
    print("Feature Engineering Completed Successfully")
    print("=" * 60)

    print(featured_df.head())

    print("\nRows :", len(featured_df))
    print("Columns :", len(featured_df.columns))


if __name__ == "__main__":
    main()