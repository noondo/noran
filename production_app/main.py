from production_app.logging_config import setup_logging, get_logger
from production_app.data import load_data, inspect_data, split_data
from production_app.training import train_pipeline
from production_app.evaluation import evaluate_pipeline
from production_app.model_service import save_pipeline, predict_sample

def run_training_workflow():
    """Orchestrate training, evaluation, and saving workflow."""
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting training workflow...")
    raw_data = load_data()
    df = raw_data.frame
    inspect_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = train_pipeline(X_train, y_train)
    evaluate_pipeline(pipeline, X_test, y_test, raw_data.target_names)
    save_pipeline(pipeline, raw_data.target_names)
    logger.info("Fitted pipeline saved successfully.")

def run_prediction_workflow():
    """Orchestrate reloading saved pipeline and predicting a sample without retraining."""
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting prediction workflow from saved artifact...")
    raw_data = load_data()
    df = raw_data.frame
    _, X_test, _, y_test = split_data(df)

    sample = X_test.iloc[[0]]
    actual_label = raw_data.target_names[y_test.iloc[0]]
    pred_label, pred_idx, confidence = predict_sample(sample)

    logger.info(
        f"Prediction -> Actual: {actual_label} | Predicted: {pred_label} ({pred_idx}) | Confidence: {confidence*100:.2f}%"
    )

def run_workflow():
    """Run both training and prediction workflows."""
    run_training_workflow()
    run_prediction_workflow()
