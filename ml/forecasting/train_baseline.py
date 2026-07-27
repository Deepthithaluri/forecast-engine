from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DATA_PATH / "daily_sales.csv",
        parse_dates=["date"],
    )


def prepare_data(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)

    df["day_number"] = range(len(df))

    X = df[["day_number"]]
    y = df["total_sales"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )


def train_model(X_train, y_train):

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("=" * 60)
    print("Baseline Linear Regression")
    print("=" * 60)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    plt.figure(figsize=(14, 6))

    plt.plot(
        y_test.values,
        label="Actual Sales",
        linewidth=2,
    )

    plt.plot(
        predictions,
        label="Predicted Sales",
        linewidth=2,
    )

    plt.title("Baseline Forecast")
    plt.xlabel("Days")
    plt.ylabel("Daily Sales")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


def main():

    df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()