from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed"

# Models
MODEL_PATH = PROJECT_ROOT / "ml" / "models"

# Recommendation model
RECOMMENDATION_MODEL_PATH = (
    MODEL_PATH / "recommendation_engine.joblib"
)