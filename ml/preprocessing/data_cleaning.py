from pathlib import Path

import pandas as pd

from ml.preprocessing.load_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the UCI Online Retail dataset.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing descriptions
    df = df.dropna(subset=["Description"])

    # Remove rows with missing customer IDs
    df = df.dropna(subset=["Customer ID"])

    # Remove cancelled invoices
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Remove invalid quantities
    df = df[df["Quantity"] > 0]

    # Remove invalid prices
    df = df[df["Price"] > 0]

    # Convert customer id to integer
    df["Customer ID"] = df["Customer ID"].astype(int)

    # Remove extra spaces
    df["Description"] = df["Description"].str.strip()

    # Sort by purchase time
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """
    Save cleaned dataset.
    """

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA_PATH / "cleaned_orders.csv",
        index=False,
    )


def main():

    df = load_dataset()

    cleaned_df = clean_data(df)

    save_processed_data(cleaned_df)

    print("=" * 60)
    print("Cleaning Completed Successfully")
    print("=" * 60)
    print(f"Rows    : {cleaned_df.shape[0]}")
    print(f"Columns : {cleaned_df.shape[1]}")

    print("\nMissing Values")

    print(cleaned_df.isnull().sum())


if __name__ == "__main__":
    main()