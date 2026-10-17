# Breast Cancer Classification — From Notebook to Production Python

A clean, reproducible Machine Learning classification project using the **Breast Cancer Wisconsin** dataset and **Logistic Regression** inside a `scikit-learn` Pipeline.

## Project Structure

```text
├── notebooks/
│   └── 01_exploration.ipynb   # Stage A: Data exploration & baseline validation
├── src/ml_project/
│   ├── __init__.py
│   ├── data.py                # Data loading, inspection, and train/test splitting
│   ├── model.py               # Pipeline creation (StandardScaler + LogisticRegression) & saving
│   ├── evaluate.py            # Model evaluation metrics
│   ├── train.py               # Training workflow entrypoint
│   └── predict.py             # Prediction workflow entrypoint
├── models/                    # Saved fitted pipeline artifacts
├── pyproject.toml             # Project dependencies managed by uv
└── uv.lock                    # Exact dependency lockfile for reproducibility