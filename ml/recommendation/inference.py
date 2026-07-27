import joblib

from ml.config import RECOMMENDATION_MODEL_PATH


def load_engine():
    """
    Load the trained recommendation engine.
    """

    return joblib.load(
        RECOMMENDATION_MODEL_PATH
    )


engine = load_engine()


def get_recommendations(
    product_name: str,
    top_n: int = 5,
):
    """
    Get recommendations for a product.
    """

    return engine.recommend(
        product_name=product_name,
        top_n=top_n,
    )


if __name__ == "__main__":

    product_name = input(
        "Enter Product Name: "
    ).strip()

    result = get_recommendations(
        product_name
    )

    print("\nRecommendations\n")

    print(f"Recommendation Type : {result['recommendation_type']}")
    print(f"Requested Product   : {result['requested_product']}")
    print(f"Recommendation Count: {result['recommendation_count']}")

    print("\nRecommended Products")
    print("-" * 60)

    if result["recommendations"]:
        for i, recommendation in enumerate(
            result["recommendations"],
            start=1,
        ):
            print(f"{i}. {recommendation['product_name']}")
            print(f"   Stock Code : {recommendation['stock_code']}")
            print(
                f"   Co-purchases : "
                f"{recommendation['co_purchase_count']}"
            )
            print()
    else:
        print("No recommendations found.")