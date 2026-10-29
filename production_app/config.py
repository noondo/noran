import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base project directory (ml project folder)
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Directories & Paths
    MODEL_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "models")
    MODELS_DIR: Path = MODEL_DIR
    PIPELINE_PATH: Path = MODELS_DIR / os.getenv("PIPELINE_FILENAME", "pipeline.joblib")
    TARGET_NAMES_PATH: Path = MODELS_DIR / os.getenv(
        "TARGET_NAMES_FILENAME", "target_names.joblib"
    )
    DATA_PATH: Path = BASE_DIR / os.getenv("DATA_PATH", "data/raw/dataset.csv")
    LOG_DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")
    LOG_FILE: Path = LOG_DIR / os.getenv("LOG_FILE", "app.log")

    # Data split & reproducibility
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", 0.2))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))

    # Model hyperparameters
    MAX_ITER: int = int(os.getenv("MAX_ITER", 1000))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 0.001))
    EPOCHS: int = int(os.getenv("EPOCHS", 10))


# Create a global configuration instance
config = Config()

# Export variables for direct imports (backward compatibility)
ENVIRONMENT = config.ENVIRONMENT
LOG_LEVEL = config.LOG_LEVEL
MODEL_DIR = config.MODEL_DIR
MODELS_DIR = config.MODELS_DIR
PIPELINE_PATH = config.PIPELINE_PATH
TARGET_NAMES_PATH = config.TARGET_NAMES_PATH
DATA_PATH = config.DATA_PATH
LOG_DIR = config.LOG_DIR
LOG_FILE = config.LOG_FILE
TEST_SIZE = config.TEST_SIZE
RANDOM_STATE = config.RANDOM_STATE
MAX_ITER = config.MAX_ITER
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS = config.EPOCHS
