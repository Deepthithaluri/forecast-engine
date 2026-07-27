from collections import Counter, defaultdict
from typing import Dict

import pandas as pd

from backend.logger import logger
from ml.config import PROCESSED_DATA_PATH
from ml.recommendation.popularity_model import get_popular_products


INVOICE = "Invoice"
STOCK_CODE = "StockCode"
DESCRIPTION = "Description"


class RecommendationEngine:
    """
    Item Co-occurrence Recommendation Engine
    for the UCI Online Retail dataset.
    """

    def __init__(self) -> None:

        self.df: pd.DataFrame | None = None

        # StockCode -> Counter(other StockCodes)
        self.cooccurrence = defaultdict(Counter)

        # StockCode -> Description
        self.product_lookup: Dict[str, str] = {}

        # Description -> StockCode
        self.description_lookup: Dict[str, str] = {}

        self.metadata: Dict[str, int | bool] = {}

        self.is_trained = False

    def load_data(self) -> None:
        """
        Load cleaned UCI dataset.
        """

        self.df = pd.read_csv(
            PROCESSED_DATA_PATH / "cleaned_orders.csv"
        )

        self.df = self.df.dropna(
            subset=[
                INVOICE,
                STOCK_CODE,
                DESCRIPTION,
            ]
        )

        self.df[DESCRIPTION] = (
            self.df[DESCRIPTION]
            .astype(str)
            .str.strip()
        )

        self.df[STOCK_CODE] = (
            self.df[STOCK_CODE]
            .astype(str)
            .str.strip()
        )

    def build_product_lookup(self) -> None:
        """
        Build lookup dictionaries.
        """

        if self.df is None:
            raise RuntimeError("Dataset not loaded.")

        lookup = (
            self.df[
                [STOCK_CODE, DESCRIPTION]
            ]
            .drop_duplicates()
        )

        self.product_lookup = (
            lookup.set_index(STOCK_CODE)[DESCRIPTION]
            .to_dict()
        )

        self.description_lookup = {
            description.lower(): stock
            for stock, description in self.product_lookup.items()
        }

    def build_cooccurrence_model(self) -> None:
        """
        Build item co-occurrence matrix using Invoice as basket.
        """

        if self.df is None:
            raise RuntimeError("Dataset not loaded.")

        baskets = (
            self.df.groupby(INVOICE)[STOCK_CODE]
            .apply(lambda x: list(set(x)))
        )

        for products in baskets:
            if len(products) < 2:
                continue

            for product in products:
                for other_product in products:

                    if product == other_product:
                        continue

                    self.cooccurrence[product][other_product] += 1

    def train(self) -> None:
        """
        Complete training pipeline.
        """

        self.load_data()

        self.build_product_lookup()

        self.build_cooccurrence_model()

        self.is_trained = True

        self.metadata = {
            "orders": int(self.df[INVOICE].nunique()),
            "products": len(self.product_lookup),
            "cooccurrence_nodes": len(self.cooccurrence),
            "trained": True,
        }

        logger.info("=" * 60)
        logger.info("Recommendation Engine Trained")
        logger.info("=" * 60)
        logger.info("Orders: %s", self.metadata["orders"])
        logger.info("Products: %s", self.metadata["products"])
        logger.info(
            "Products With Recommendations: %s",
            self.metadata["cooccurrence_nodes"],
        )
        logger.info("=" * 60)

    def recommend(
        self,
        product_name: str,
        top_n: int = 5,
    ) -> dict:
        """
        Recommend products using product description.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Recommendation engine is not trained."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        stock_code = self.description_lookup.get(
            product_name.lower().strip()
        )

        # Popularity fallback
        if stock_code is None or stock_code not in self.cooccurrence:

            popular_products = get_popular_products(top_n)

            recommendations = [
                {
                    "stock_code": row["StockCode"],
                    "product_name": row["Description"],
                    "co_purchase_count": int(
                        row["purchase_count"]
                    ),
                }
                for _, row in popular_products.iterrows()
            ]

            return {
                "recommendation_type": "popularity_fallback",
                "requested_product": product_name,
                "recommendation_count": len(recommendations),
                "recommendations": recommendations,
            }

        similar_products = (
            self.cooccurrence[stock_code]
            .most_common(top_n)
        )

        recommendation_list = []

        for code, count in similar_products:

            recommendation_list.append(
                {
                    "stock_code": code,
                    "product_name": self.product_lookup.get(
                        code,
                        "Unknown Product",
                    ),
                    "co_purchase_count": count,
                }
            )

        return {
            "recommendation_type": "cooccurrence",
            "requested_product": product_name,
            "recommendation_count": len(
                recommendation_list
            ),
            "recommendations": recommendation_list,
        }

    def get_training_summary(self) -> dict:

        if not self.is_trained:
            raise RuntimeError(
                "Recommendation engine is not trained."
            )

        return self.metadata


if __name__ == "__main__":

    engine = RecommendationEngine()

    engine.train()

    print("\nTraining Summary\n")
    print(engine.get_training_summary())

    sample_product = next(
        iter(engine.product_lookup.values())
    )

    print(f"\nSample Product:\n{sample_product}")

    recommendations = engine.recommend(sample_product)

    print("\nRecommendations\n")
    print(recommendations)