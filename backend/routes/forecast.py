from fastapi import APIRouter

from backend.schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
)
from backend.services.forecasting_service import (
    generate_forecast,
)

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"],
)


@router.post(
    "/",
    response_model=ForecastResponse,
)
def get_forecast(
    request: ForecastRequest,
):
    return generate_forecast(request)