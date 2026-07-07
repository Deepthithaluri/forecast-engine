from fastapi import FastAPI

from backend.routes.health import router as health_router

app = FastAPI(
    title="E-Commerce Sales Forecasting API",
    description="Backend API for sales forecasting and product recommendations",
    version="1.0.0",
)

app.include_router(health_router)