from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    product_id: str
    top_n: int = 5


class RecommendationItem(BaseModel):
    product_id: str
    category: str
    co_purchase_count: int


class RecommendationResponse(BaseModel):
    recommendation_type: str
    requested_product: str
    recommendation_count: int
    recommendations: list[RecommendationItem]