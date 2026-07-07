import joblib

from ml.config import RECOMMENDATION_MODEL_PATH


def evaluate() -> None:
    """
    Evaluate the saved recommendation engine.
    """

    engine = joblib.load(RECOMMENDATION_MODEL_PATH)

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

    sample_product = next(iter(engine.cooccurrence.keys()))

    print(f"\nSample Product:\n{sample_product}")

    recommendations = engine.recommend(sample_product)

    print("\nRecommendations\n")

    if recommendations:
        for recommendation in recommendations:
            print(recommendation)
    else:
        print("No recommendations found.")


if __name__ == "__main__":
    evaluate()