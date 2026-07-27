from pathlib import Path

import pandas as pd

from ml.config import RAW_DATA_PATH


EXCEL_FILE = "Online Retail.xlsx"


def load_dataset() -> pd.DataFrame:
    """
    Load both sheets from the UCI Online Retail dataset
    and combine them into a single DataFrame.
    """

    file_path = RAW_DATA_PATH / EXCEL_FILE

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    year_2009 = pd.read_excel(
        file_path,
        sheet_name="Year 2009-2010",
    )

    year_2010 = pd.read_excel(
        file_path,
        sheet_name="Year 2010-2011",
    )

    df = pd.concat(
        [year_2009, year_2010],
        ignore_index=True,
    )

    return df


def display_basic_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("UCI Online Retail Dataset")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nFirst Five Rows")
    print(df.head())


def main():
    df = load_dataset()
    display_basic_info(df)


if __name__ == "__main__":
    main()