import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MODELS_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
TARGET_NAMES_PATH = MODELS_DIR / "target_names.joblib"

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MAX_ITER = 1000
