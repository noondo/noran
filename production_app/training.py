from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from production_app.config import RANDOM_STATE, MAX_ITER
from production_app.preprocessing import create_preprocessor

def train_pipeline(X_train, y_train):
    """Create model, combine with preprocessing in a Pipeline, and fit on training data."""
    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
