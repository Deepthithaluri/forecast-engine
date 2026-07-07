from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "xgboost_sales_forecasting.pkl"
)

model = joblib.load(MODEL_PATH)

def generate_forecast():
    future_data = pd.DataFrame(
        {
            "lag_1": [1500],
            "lag_7": [1450],
            "rolling_mean_7": [1480],
            "rolling_std_7": [120],
            "month": [7],
            "day": [7],
            "weekday": [1],
        }
    )

    prediction = model.predict(future_data)

    return [
        {
            "predicted_sales": round(float(prediction[0]), 2)
        }
    ]