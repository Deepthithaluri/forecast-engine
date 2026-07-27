from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "ml" / "models"


MODEL_FILE = MODEL_PATH / "xgboost_sales_forecasting.pkl"


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
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    plt.figure(figsize=(15, 6))

    plt.plot(
        y_test.values,
        label="Actual",
        linewidth=2,
    )

    plt.plot(
        predictions,
        label="Predicted",
        linewidth=2,
    )

    plt.title("Sales Forecasting using XGBoost")
    plt.xlabel("Days")
    plt.ylabel("Daily Sales")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model(model):

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved successfully:\n{MODEL_FILE}")


def main():

    print("=" * 60)
    print("Training XGBoost Forecasting Model")
    print("=" * 60)

    df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(X_train, y_train)

    save_model(model)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()