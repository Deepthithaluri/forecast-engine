from fastapi import FastAPI

from backend.exceptions import (
    register_exception_handlers,
)
from backend.logger import logger
from backend.middleware import (
    register_middlewares,
)
from backend.routes.forecast import (
    router as forecast_router,
)
from backend.routes.health import (
    router as health_router,
)
from backend.routes.recommendation import (
    router as recommendation_router,
)
from ml.recommendation.popularity_model import (
    load_popular_products,
)

app = FastAPI(
    title="E-Commerce Sales Forecasting API",
    description=(
        "Backend API for sales forecasting "
        "and product recommendations"
    ),
    version="1.0.0",
)

logger.info("Application started successfully.")

load_popular_products()

logger.info("Popular products cache loaded.")

register_middlewares(app)
register_exception_handlers(app)

app.include_router(health_router)
app.include_router(forecast_router)
app.include_router(recommendation_router)