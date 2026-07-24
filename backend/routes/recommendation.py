from fastapi import APIRouter

from backend.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from backend.services.recommendation_service import (
    generate_recommendations,
)

router = APIRouter(
    prefix="/recommendation",
    tags=["Recommendation"],
)


@router.post(
    "/",
    response_model=RecommendationResponse,
)
def recommend_products(
    request: RecommendationRequest,
):
    return generate_recommendations(
        product_id=request.product_id,
        top_n=request.top_n,
    )