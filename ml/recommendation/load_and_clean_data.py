from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("ml/data/raw/instacart")
OUTPUT_DIR = Path("ml/data/processed")

ORDERS_FILE = RAW_DATA_DIR / "orders.csv"
PRIOR_FILE = RAW_DATA_DIR / "order_products__prior.csv"
TRAIN_FILE = RAW_DATA_DIR / "order_products__train.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
AISLES_FILE = RAW_DATA_DIR / "aisles.csv"
DEPARTMENTS_FILE = RAW_DATA_DIR / "departments.csv"

OUTPUT_FILE = OUTPUT_DIR / "recommendation_cleaned.csv"


def load_csv(file_path: Path, name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    df = pd.read_csv(file_path)

    logger.info("Loaded %s (%d rows)", name, len(df))

    return df


def validate_dataframes(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
) -> None:

    required = {
        "orders": ["order_id", "user_id"],
        "order_products": ["order_id", "product_id"],
        "products": ["product_id", "product_name", "aisle_id", "department_id"],
        "aisles": ["aisle_id", "aisle"],
        "departments": ["department_id", "department"],
    }

    datasets = {
        "orders": orders,
        "order_products": order_products,
        "products": products,
        "aisles": aisles,
        "departments": departments,
    }

    for name, columns in required.items():
        missing = set(columns) - set(datasets[name].columns)

        if missing:
            raise ValueError(f"{name} missing columns: {missing}")


def merge_data(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
) -> pd.DataFrame:

    df = order_products.merge(
        orders,
        on="order_id",
        how="left",
    )

    df = df.merge(
        products,
        on="product_id",
        how="left",
    )

    df = df.merge(
        aisles,
        on="aisle_id",
        how="left",
    )

    df = df.merge(
        departments,
        on="department_id",
        how="left",
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Cleaning merged dataset...")

    df = df.drop_duplicates()

    df = df.dropna(
        subset=[
            "user_id",
            "product_id",
            "product_name",
        ]
    )

    df["product_name"] = (
        df["product_name"]
        .astype(str)
        .str.strip()
    )

    return df


def save_data(df: pd.DataFrame) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info("Saved cleaned dataset to %s", OUTPUT_FILE)


def print_summary(df: pd.DataFrame) -> None:

    print("\nRecommendation Dataset Summary\n")

    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns)}")
    print(f"Users             : {df['user_id'].nunique():,}")
    print(f"Products          : {df['product_id'].nunique():,}")
    print(f"Orders            : {df['order_id'].nunique():,}")
    print(f"Departments       : {df['department'].nunique()}")
    print(f"Aisles            : {df['aisle'].nunique()}")


def main() -> None:

    orders = load_csv(
        ORDERS_FILE,
        "orders",
    )

    prior = load_csv(
        PRIOR_FILE,
        "order_products__prior",
    )

    train = load_csv(
        TRAIN_FILE,
        "order_products__train",
    )

    order_products = pd.concat(
        [
            prior,
            train,
        ],
        ignore_index=True,
    )

    logger.info(
        "Combined order products (%d rows)",
        len(order_products),
    )

    products = load_csv(
        PRODUCTS_FILE,
        "products",
    )

    aisles = load_csv(
        AISLES_FILE,
        "aisles",
    )

    departments = load_csv(
        DEPARTMENTS_FILE,
        "departments",
    )

    validate_dataframes(
        orders,
        order_products,
        products,
        aisles,
        departments,
    )

    recommendation_df = merge_data(
        orders,
        order_products,
        products,
        aisles,
        departments,
    )

    recommendation_df = clean_data(
        recommendation_df,
    )

    save_data(
        recommendation_df,
    )

    print_summary(
        recommendation_df,
    )


if __name__ == "__main__":
    main()