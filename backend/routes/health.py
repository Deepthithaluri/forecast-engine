from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Welcome to the E-Commerce Sales Forecasting API",
        "status": "running",
    }


@router.get("/health")
def health():
    return {
        "status": "healthy",
    }