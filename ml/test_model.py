from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "xgboost_sales_forecasting.pkl"

print("Model path:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

model = joblib.load(MODEL_PATH)

print("Loaded successfully!")