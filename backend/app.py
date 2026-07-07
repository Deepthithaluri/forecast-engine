from fastapi import FastAPI

from backend.routes.forecast import router as forecast_router
from backend.routes.health import router as health_router

app = FastAPI(
    title="E-Commerce Sales Forecasting API",
    description="Backend API for sales forecasting and product recommendations",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(forecast_router)