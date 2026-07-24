from ml.recommendation.inference import (
    get_recommendations,
)


def generate_recommendations(
    product_id: str,
    top_n: int,
):
    return get_recommendations(
        product_id=product_id,
        top_n=top_n,
    )