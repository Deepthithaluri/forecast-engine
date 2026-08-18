from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

MODEL_FILE = Path("ml/models/forecasting_model.joblib")
INPUT_FILE = Path("ml/data/processed/daily_sales.csv")

OUTPUT_DIR = Path("ml/predictions")
OUTPUT_FILE = OUTPUT_DIR / "sales_forecast.csv"

FORECAST_DAYS = 7


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"{MODEL_FILE} not found.")

    logger.info("Loading forecasting model")

    return joblib.load(MODEL_FILE)


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Order Date"],
    )

    logger.info("Loaded %d daily observations", len(df))

    return df


def create_feature_row(history: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    sales = history["daily_sales"]

    row = {
        "total_orders": history["total_orders"].iloc[-1],
        "total_quantity": history["total_quantity"].iloc[-1],
        "total_profit": history["total_profit"].iloc[-1],
        "year": forecast_date.year,
        "month": forecast_date.month,
        "day": forecast_date.day,
        "day_of_week": forecast_date.dayofweek,
        "is_weekend": int(forecast_date.dayofweek >= 5),
        "lag_1": sales.iloc[-1],
        "lag_7": sales.iloc[-7],
        "lag_14": sales.iloc[-14],
        "lag_28": sales.iloc[-28],
        "rolling_mean_7": sales.iloc[-7:].mean(),
        "rolling_std_7": sales.iloc[-7:].std(),
        "rolling_mean_14": sales.iloc[-14:].mean(),
        "rolling_std_14": sales.iloc[-14:].std(),
        "rolling_mean_28": sales.iloc[-28:].mean(),
        "rolling_std_28": sales.iloc[-28:].std(),
        "expanding_mean": sales.mean(),
        "expanding_std": sales.std(),
        "day_of_week_sin": np.sin(2 * np.pi * forecast_date.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * forecast_date.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * forecast_date.month / 12),
        "month_cos": np.cos(2 * np.pi * forecast_date.month / 12),
    }

    return pd.DataFrame([row])


def forecast(model, history: pd.DataFrame) -> pd.DataFrame:
    predictions = []

    history = history.copy()

    for _ in range(FORECAST_DAYS):

        next_date = history["Order Date"].max() + pd.Timedelta(days=1)

        X = create_feature_row(history, next_date)

        predicted_sales = float(model.predict(X)[0])

        predictions.append(
            {
                "Order Date": next_date,
                "Predicted Sales": round(predicted_sales, 2),
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "Order Date": [next_date],
                        "daily_sales": [predicted_sales],
                        "total_orders": [history["total_orders"].iloc[-1]],
                        "total_quantity": [history["total_quantity"].iloc[-1]],
                        "total_profit": [history["total_profit"].iloc[-1]],
                    }
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


def save_forecast(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info(
        "Saved forecast to %s",
        OUTPUT_FILE,
    )


def print_summary(df: pd.DataFrame) -> None:
    print("\nSales Forecast\n")

    print(df.to_string(index=False))

    print(f"\nForecast Horizon : {len(df)} days")
    print(f"Total Forecast   : ₹{df['Predicted Sales'].sum():,.2f}")
    print(f"Average Forecast : ₹{df['Predicted Sales'].mean():,.2f}")


def main() -> None:
    model = load_model()

    history = load_data()

    forecast_df = forecast(
        model,
        history,
    )

    save_forecast(
        forecast_df,
    )

    print_summary(
        forecast_df,
    )


if __name__ == "__main__":
    main()