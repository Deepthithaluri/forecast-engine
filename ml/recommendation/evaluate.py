import joblib

from ml.config import RECOMMENDATION_MODEL_PATH


def evaluate() -> None:
    """
    Evaluate the saved recommendation engine.
    """

    engine = joblib.load(
        RECOMMENDATION_MODEL_PATH
    )

    print("=" * 60)
    print("Recommendation Engine Evaluation")
    print("=" * 60)

    summary = engine.get_training_summary()

    print("\nTraining Summary\n")

    for key, value in summary.items():
        print(f"{key:<25}: {value}")

    print(
        f"\nProducts with recommendation data: "
        f"{len(engine.cooccurrence)}"
    )

    if not engine.cooccurrence:
        print("No recommendation data available.")
        return

    # Get a sample product description instead of StockCode
    sample_product = next(
        iter(engine.product_lookup.values())
    )

    print(f"\nSample Product:\n{sample_product}")

    result = engine.recommend(
        sample_product,
        top_n=5,
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


if __name__ == "__main__":
    evaluate()