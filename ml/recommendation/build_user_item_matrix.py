from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

INPUT_FILE = Path("ml/data/processed/recommendation_cleaned.csv")

OUTPUT_DIR = Path("ml/data/processed")
OUTPUT_FILE = OUTPUT_DIR / "user_product_interactions.csv"

REQUIRED_COLUMNS = [
    "user_id",
    "product_id",
    "product_name",
]


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=REQUIRED_COLUMNS,
    )

    logger.info(
        "Loaded %s interaction records",
        f"{len(df):,}",
    )

    return df


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input dataset is empty.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building user-product interactions...")

    interactions = (
        df.groupby(
            [
                "user_id",
                "product_id",
                "product_name",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "purchase_count",
            }
        )
        .sort_values(
            [
                "user_id",
                "purchase_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    return interactions


def save_data(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info(
        "Saved user-product interactions to %s",
        OUTPUT_FILE,
    )


def print_summary(df: pd.DataFrame) -> None:
    print("\nUser-Product Interaction Summary\n")

    print(f"Interaction Records : {len(df):,}")
    print(f"Unique Users        : {df['user_id'].nunique():,}")
    print(f"Unique Products     : {df['product_id'].nunique():,}")
    print(f"Average Purchases   : {df['purchase_count'].mean():.2f}")
    print(f"Maximum Purchases   : {df['purchase_count'].max()}")


def main() -> None:
    df = load_data()

    validate_data(df)

    interactions = build_interactions(df)

    save_data(interactions)

    print_summary(interactions)


if __name__ == "__main__":
    main()