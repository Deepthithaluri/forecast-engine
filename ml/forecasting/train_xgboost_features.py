from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        PROCESSED_DATA_PATH / "daily_sales_features.csv",
        parse_dates=["date"],
    )


def prepare_data(df: pd.DataFrame):
    features = [
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "month",
        "day",
        "weekday",
    ]

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[features]
    y_train = train_df["total_sales"]

    X_test = test_df[features]
    y_test = test_df["total_sales"]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    plt.figure(figsize=(14, 6))

    plt.plot(y_test.values, label="Actual Sales", linewidth=2)
    plt.plot(predictions, label="Predicted Sales", linewidth=2)

    plt.title("XGBoost with Time-Series Features")
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