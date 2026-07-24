from pathlib import Path

import joblib
import pandas as pd

from backend.schemas.forecast import ForecastRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "xgboost_sales_forecasting.pkl"
)

model = joblib.load(MODEL_PATH)


def generate_forecast(request: ForecastRequest) -> float:
    future_data = pd.DataFrame(
        {
            "lag_1": [request.lag_1],
            "lag_7": [request.lag_7],
            "rolling_mean_7": [request.rolling_mean_7],
            "rolling_std_7": [request.rolling_std_7],
            "month": [request.month],
            "day": [request.day],
            "weekday": [request.weekday],
        }
    )

    prediction = model.predict(future_data)

    return round(float(prediction[0]), 2)