from collections import Counter, defaultdict
from typing import Dict

import pandas as pd

from ml.config import PROCESSED_DATA_PATH
from ml.recommendation.popularity_model import (
    get_popular_products,
)

# Constants
ORDER_ID = "order_id"
PRODUCT_ID = "product_id"
STATUS = "order_status"
DELIVERED = "delivered"
CATEGORY_EN = "product_category_name_english"
CATEGORY = "product_category_name"


class RecommendationEngine:
    """
    Item Co-occurrence Recommendation Engine.
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None

        self.cooccurrence: defaultdict = defaultdict(Counter)

        self.product_lookup: Dict[str, str] = {}

        self.metadata: Dict[str, int | bool] = {}

        self.is_trained = False

    def load_data(self) -> None:
        """
        Load the processed dataset.
        """

        self.df = pd.read_csv(
            PROCESSED_DATA_PATH / "integrated_dataset.csv"
        )

        self.df = self.df[
            self.df[STATUS] == DELIVERED
        ]

        self.df = self.df.dropna(
            subset=[
                ORDER_ID,
                PRODUCT_ID,
            ]
        )

    def build_product_lookup(self) -> None:
        """
        Build Product → Category lookup.
        """

        if self.df is None:
            raise RuntimeError("Dataset not loaded.")

        category_column = (
            CATEGORY_EN
            if CATEGORY_EN in self.df.columns
            else CATEGORY
        )

        self.product_lookup = (
            self.df[
                [
                    PRODUCT_ID,
                    category_column,
                ]
            ]
            .drop_duplicates()
            .set_index(PRODUCT_ID)[category_column]
            .fillna("Unknown")
            .to_dict()
        )

    def build_cooccurrence_model(self) -> None:
        """
        Build item co-occurrence matrix.
        """

        if self.df is None:
            raise RuntimeError("Dataset not loaded.")

        order_products = (
            self.df.groupby(ORDER_ID)[PRODUCT_ID]
            .apply(list)
        )

        for products in order_products:

            unique_products = list(set(products))

            if len(unique_products) < 2:
                continue

            for product in unique_products:

                for other_product in unique_products:

                    if product == other_product:
                        continue

                    self.cooccurrence[product][
                        other_product
                    ] += 1

    def train(self) -> None:
        """
        Complete training pipeline.
        """

        self.load_data()

        self.build_product_lookup()

        self.build_cooccurrence_model()

        self.is_trained = True

        self.metadata = {
            "orders": int(
                self.df[ORDER_ID].nunique()
            ),
            "products": len(
                self.product_lookup
            ),
            "cooccurrence_nodes": len(
                self.cooccurrence
            ),
            "trained": True,
        }

        print("=" * 60)
        print("Recommendation Engine Trained")
        print("=" * 60)

        print(
            f"Orders                : {self.metadata['orders']}"
        )

        print(
            f"Products              : {self.metadata['products']}"
        )

        print(
            "Products With "
            f"Recommendations : {self.metadata['cooccurrence_nodes']}"
        )

        print("=" * 60)

    def recommend(
        self,
        product_id: str,
        top_n: int = 5,
    ) -> dict:
        """
        Recommend similar products.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Recommendation engine is not trained."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        # Cold-start fallback
        if product_id not in self.cooccurrence:

            popular_products = get_popular_products(top_n)

            recommendations = [
                {
                    "product_id": row["product_id"],
                    "category": row[
                        "product_category_name_english"
                    ],
                    "co_purchase_count": int(
                        row["purchase_count"]
                    ),
                }
                for _, row in popular_products.iterrows()
            ]

            return {
                "recommendation_type": "popularity_fallback",
                "requested_product": product_id,
                "recommendation_count": len(recommendations),
                "recommendations": recommendations,
            }

        recommendations = (
            self.cooccurrence[product_id]
            .most_common(top_n)
        )

        recommendation_list = [
            {
                "product_id": product,
                "category": self.product_lookup.get(
                    product,
                    "Unknown",
                ),
                "co_purchase_count": count,
            }
            for product, count in recommendations
        ]

        return {
            "recommendation_type": "cooccurrence",
            "requested_product": product_id,
            "recommendation_count": len(
                recommendation_list
            ),
            "recommendations": recommendation_list,
        }

    def get_training_summary(self) -> dict:
        """
        Return training statistics.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Recommendation engine is not trained."
            )

        return self.metadata


if __name__ == "__main__":

    engine = RecommendationEngine()

    engine.train()

    print("\nMetadata\n")

    print(engine.metadata)

    if engine.cooccurrence:

        sample_product = next(
            iter(engine.cooccurrence.keys())
        )

        print(
            f"\nSample Product:\n{sample_product}"
        )

        recommendations = engine.recommend(
            sample_product
        )

        print("\nRecommendations\n")

        print(recommendations)