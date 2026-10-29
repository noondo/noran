from pathlib import Path
from typing import Any

import joblib

from production_app.config import PIPELINE_PATH, TARGET_NAMES_PATH


# ---------------------------------------------------------
# 1.  (Training & Saving)
# ---------------------------------------------------------
def save_pipeline(pipeline: Any, target_names: Any) -> None:
    """Saves the trained pipeline and target names to disk."""

    Path(PIPELINE_PATH).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(target_names, TARGET_NAMES_PATH)


def predict_sample(model: Any, input_data: Any) -> Any:
    """Predicts a single sample (used during training evaluation)."""
    return model.predict([input_data])[0]


# ---------------------------------------------------------
# 2.(API Loading & Serving)
# ---------------------------------------------------------
def load_trained_model(model_path: str | Path) -> Any:
    """Loads the trained model for the API."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Please train the model first by running: uv run python run.py train"
        )
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from {path}: {e}")


def predict(model: Any, input_data: list[float]) -> Any:
    """Performs inference on incoming API requests."""
    if not input_data:
        raise ValueError("Input data for prediction cannot be empty.")
    prediction = model.predict([input_data])
    return prediction[0]
