from pathlib import Path
import logging

import joblib
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

INPUT_FILE = Path("ml/data/processed/user_product_interactions.csv")

MODEL_DIR = Path("ml/models")

MODEL_FILE = MODEL_DIR / "recommendation_model.joblib"
PRODUCT_INDEX_FILE = MODEL_DIR / "product_index_mapping.joblib"
PRODUCT_CATALOG_FILE = MODEL_DIR / "product_catalog.joblib"

REQUIRED_COLUMNS = [
    "user_id",
    "product_id",
    "product_name",
    "purchase_count",
]


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_csv(INPUT_FILE)

    logger.info(
        "Loaded %s interaction records",
        f"{len(df):,}",
    )

    return df


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Interaction dataset is empty.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    logger.info("Input validation passed.")


def build_sparse_matrix(
    df: pd.DataFrame,
) -> tuple[csr_matrix, dict, pd.DataFrame]:

    logger.info("Creating user and product indices...")

    user_ids = sorted(
        df["user_id"].unique()
    )

    product_catalog = (
        df[
            [
                "product_id",
                "product_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    user_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    product_index = {
        product_id: index
        for index, product_id in enumerate(
            product_catalog["product_id"]
        )
    }

    logger.info("Building sparse user-item matrix...")

    rows = df["user_id"].map(user_index)
    cols = df["product_id"].map(product_index)
    data = df["purchase_count"]

    sparse_matrix = csr_matrix(
        (
            data,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            len(user_ids),
            len(product_catalog),
        ),
    )

    logger.info(
        "Sparse matrix created with shape %s",
        sparse_matrix.shape,
    )

    return (
        sparse_matrix,
        product_index,
        product_catalog,
    )


def train_model(
    sparse_matrix: csr_matrix,
) -> NearestNeighbors:

    logger.info("Training recommendation model...")

    item_matrix = sparse_matrix.T

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=20,
        n_jobs=-1,
    )

    model.fit(item_matrix)

    logger.info("Model training completed.")

    return model


def save_artifacts(
    model: NearestNeighbors,
    product_index: dict,
    product_catalog: pd.DataFrame,
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        product_index,
        PRODUCT_INDEX_FILE,
    )

    joblib.dump(
        product_catalog,
        PRODUCT_CATALOG_FILE,
    )

    logger.info(
        "Saved recommendation model to %s",
        MODEL_FILE,
    )

    logger.info(
        "Saved product index mapping to %s",
        PRODUCT_INDEX_FILE,
    )

    logger.info(
        "Saved product catalog to %s",
        PRODUCT_CATALOG_FILE,
    )


def print_summary(
    interactions: pd.DataFrame,
    sparse_matrix: csr_matrix,
) -> None:

    print("\nRecommendation Model Training Summary\n")

    print(f"Interaction Records : {len(interactions):,}")
    print(f"Users               : {sparse_matrix.shape[0]:,}")
    print(f"Products            : {sparse_matrix.shape[1]:,}")
    print(f"Interactions        : {sparse_matrix.nnz:,}")
    print("Algorithm           : Item-Based Collaborative Filtering")
    print("Similarity Metric   : Cosine Distance")
    print("Nearest Neighbors   : 20")


def main() -> None:

    interactions = load_data()

    validate_data(interactions)

    sparse_matrix, product_index, product_catalog = (
        build_sparse_matrix(interactions)
    )

    model = train_model(
        sparse_matrix,
    )

    save_artifacts(
        model,
        product_index,
        product_catalog,
    )

    print_summary(
        interactions,
        sparse_matrix,
    )


if __name__ == "__main__":
    main()