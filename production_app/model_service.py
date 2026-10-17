import joblib
from production_app.config import MODELS_DIR, PIPELINE_PATH, TARGET_NAMES_PATH

def save_pipeline(pipeline, target_names):
    """Save the fitted pipeline artifact with joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(target_names, TARGET_NAMES_PATH)

def load_pipeline():
    """Load the saved pipeline artifact."""
    pipeline = joblib.load(PIPELINE_PATH)
    target_names = joblib.load(TARGET_NAMES_PATH)
    return pipeline, target_names

def predict_sample(sample_df):
    """Predict using the saved artifact without retraining."""
    pipeline, target_names = load_pipeline()
    pred_class = pipeline.predict(sample_df)[0]
    pred_prob = pipeline.predict_proba(sample_df)[0]
    return str(target_names[pred_class]), int(pred_class), float(pred_prob[pred_class])
