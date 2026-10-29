from fastapi import FastAPI, HTTPException, status
from pathlib import Path
import joblib

from production_app.config import PIPELINE_PATH, TARGET_NAMES_PATH
from production_app.model_service import load_trained_model
from production_app.logging_config import setup_logging, get_logger
from production_app.schemas import PredictionRequest, PredictionResponse

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="ML Model Serving API",
    description="API for serving the machine learning classification pipeline",
    version="2.0.0"
)
pipeline = None
target_names = []

@app.on_event("startup")
def startup_event():
    global pipeline, target_names
    try:
        pipeline = load_trained_model(PIPELINE_PATH)
        if Path(TARGET_NAMES_PATH).exists():
            target_names = joblib.load(TARGET_NAMES_PATH)
        else:
            target_names = ["malignant", "benign"]
        logger.info("Pipeline and target names loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load model artifacts on startup: {e}")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Return a meaningful JSON status distinguishing healthy service from missing model."""
    if pipeline is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "message": "Service is running, but the model artifact is missing or failed to load."
        }
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "Service is running and model is loaded successfully."
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_endpoint(request: PredictionRequest):
    """Accept one observation, run inference using saved pipeline, and return prediction."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable. Please train and save the model first."
        )
    
    try:
        # ترتيب الخصائص الـ 30
        input_features = [[
            request.mean_radius, request.mean_texture, request.mean_perimeter, request.mean_area,
            request.mean_smoothness, request.mean_compactness, request.mean_concavity, request.mean_concave_points,
            request.mean_symmetry, request.mean_fractal_dimension, request.radius_error, request.texture_error,
            request.perimeter_error, request.area_error, request.smoothness_error, request.compactness_error,
            request.concavity_error, request.concave_points_error, request.symmetry_error, request.fractal_dimension_error,
            request.worst_radius, request.worst_texture, request.worst_perimeter, request.worst_area,
            request.worst_smoothness, request.worst_compactness, request.worst_concavity, request.worst_concave_points,
            request.worst_symmetry, request.worst_fractal_dimension
        ]]
        
        prediction_idx = int(pipeline.predict(input_features)[0])
        
        confidence = None
        if hasattr(pipeline, "predict_proba"):
            import numpy as np
            probas = pipeline.predict_proba(input_features)
            confidence = float(np.max(probas) * 100)

        label = target_names[prediction_idx] if prediction_idx < len(target_names) else str(prediction_idx)

        return PredictionResponse(
            predicted_class=prediction_idx,
            label=label,
            confidence_score=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed request or prediction failed. Please check your input data."
        )