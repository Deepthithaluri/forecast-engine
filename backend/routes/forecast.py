from fastapi import APIRouter

from backend.services.forecasting_service import generate_forecast

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"],
)


@router.get("/")
def get_forecast():
    return {
        "model": "XGBoost",
        "forecast": generate_forecast(),
    }