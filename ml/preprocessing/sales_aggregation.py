from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load cleaned UCI Online Retail dataset.
    """
    return pd.read_csv(
        PROCESSED_DATA_PATH / "cleaned_orders.csv",
        parse_dates=["InvoiceDate"],
    )


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales.
    """

    # Sales value for each transaction
    df["Sales"] = df["Quantity"] * df["Price"]

    daily_sales = (
        df.groupby(df["InvoiceDate"].dt.date)
        .agg(
            total_sales=("Sales", "sum"),
            total_orders=("Invoice", "nunique"),
            total_items=("Quantity", "sum"),
        )
        .reset_index()
    )

    daily_sales.rename(
        columns={"InvoiceDate": "date"},
        inplace=True,
    )

    daily_sales["date"] = pd.to_datetime(daily_sales["date"])

    return daily_sales


def save_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA_PATH / "daily_sales.csv",
        index=False,
    )


def main():

    df = load_cleaned_dataset()

    daily_sales = aggregate_daily_sales(df)

    save_dataset(daily_sales)

    print("=" * 60)
    print("Daily Sales Dataset Created Successfully")
    print("=" * 60)
    print(daily_sales.head())

    print("\nRows :", len(daily_sales))


if __name__ == "__main__":
    main()