from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    lag_1: float = Field(..., description="Previous day's sales")
    lag_7: float = Field(..., description="Sales from seven days ago")
    rolling_mean_7: float = Field(..., description="7-day rolling average")
    rolling_std_7: float = Field(..., description="7-day rolling standard deviation")
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    weekday: int = Field(..., ge=0, le=6)


class ForecastResponse(BaseModel):
    model: str
    predicted_sales: float