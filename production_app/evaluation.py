from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from production_app.logging_config import get_logger

logger = get_logger(__name__)

def evaluate_pipeline(pipeline, X_test, y_test, target_names):
    """Calculate and report classification metrics and confusion matrix."""
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    logger.info(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    return acc
