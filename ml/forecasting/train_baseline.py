from pathlib import Path

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DATA_PATH / "daily_sales.csv",
        parse_dates=["date"],
    )


def prepare_data(df: pd.DataFrame):
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
    import matplotlib.pyplot as plt

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")

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

    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Days")
    plt.ylabel("Sales")

    plt.legend()

    plt.grid(True)

    plt.show()


def main():
    df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()