from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_and_fit_pipeline(X_train, y_train, random_state=42):
    """4. Preprocess -> 5. Fit model inside a single scikit-learn Pipeline"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
    ])
    
    # الـ Pipeline هنا بتعمل fit للـ scaler على X_train بس (تمنع Data Leakage) وبعدين تدرب الموديل
    pipeline.fit(X_train, y_train)
    return pipeline

def save_pipeline(pipeline, target_names, output_dir="models"):
    """7. Save fitted pipeline"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path / "pipeline.joblib")
    joblib.dump(target_names, path / "target_names.joblib")

def load_pipeline(model_dir="models"):
    """8. Reload fitted pipeline"""
    path = Path(model_dir)
    pipeline = joblib.load(path / "pipeline.joblib")
    target_names = joblib.load(path / "target_names.joblib")
    return pipeline, target_names