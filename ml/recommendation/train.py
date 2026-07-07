import joblib

from ml.config import (
    MODEL_PATH,
    RECOMMENDATION_MODEL_PATH,
)
from ml.recommendation.recommendation_engine import (
    RecommendationEngine,
)


def train() -> None:
    """
    Train and save the recommendation engine.
    """

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = RecommendationEngine()

    engine.train()

    joblib.dump(
        engine,
        RECOMMENDATION_MODEL_PATH,
    )

    print("\nRecommendation model saved successfully!")

    print(
        f"Location: {RECOMMENDATION_MODEL_PATH}"
    )


if __name__ == "__main__":
    train()