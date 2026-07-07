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
    product_id: str,
    top_n: int = 5,
):
    """
    Get recommendations for a product.
    """

    return engine.recommend(
        product_id=product_id,
        top_n=top_n,
    )


if __name__ == "__main__":

    product_id = input(
        "Enter Product ID: "
    ).strip()

    recommendations = get_recommendations(
        product_id
    )

    print("\nRecommendations\n")

    if recommendations:
        for recommendation in recommendations:
            print(recommendation)
    else:
        print("No recommendations found.")