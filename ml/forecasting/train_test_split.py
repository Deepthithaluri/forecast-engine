from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

INPUT_FILE = Path("ml/data/processed/training_dataset.csv")
OUTPUT_DIR = Path("ml/data/processed")

TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

TRAIN_RATIO = 0.8


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_csv(INPUT_FILE, parse_dates=["Order Date"])

    logger.info("Loaded %d rows", len(df))

    return df


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df["Order Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    if not df["Order Date"].is_monotonic_increasing:
        raise ValueError("Dataset is not sorted by date.")


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def save_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    logger.info("Saved train dataset to %s", TRAIN_FILE)
    logger.info("Saved test dataset to %s", TEST_FILE)


def print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\nTrain/Test Split Summary")
    print(f"Training Rows : {len(train_df)}")
    print(f"Testing Rows  : {len(test_df)}")

    print(f"\nTrain Range   : {train_df['Order Date'].min().date()} -> {train_df['Order Date'].max().date()}")
    print(f"Test Range    : {test_df['Order Date'].min().date()} -> {test_df['Order Date'].max().date()}")


def main() -> None:
    df = load_data()

    validate_data(df)

    train_df, test_df = split_dataset(df)

    save_dataset(train_df, test_df)

    print_summary(train_df, test_df)


if __name__ == "__main__":
    main()