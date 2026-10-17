# Breast Cancer Classification — Production Python Pipeline

A modular, robust Machine Learning classification project refactored from an exploratory Jupyter Notebook into a clean, reproducible Python application using **scikit-learn**, **uv**, and enterprise engineering standards.

##  Project Goal & Scope
The objective of this project is to build, validate, serialize, and deploy a machine learning classification model following strict software engineering principles:
* Elimination of hardcoded variables through centralized configuration.
* Prevention of Data Leakage using `scikit-learn` Pipelines.
* Comprehensive operational logging instead of standard print statements.
* Graceful exception handling for artifacts and user inputs.
* Complete environment reproducibility via `uv`.

##  Dataset & Model Choice
* **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset, loaded directly via `scikit-learn` (`load_breast_cancer`). It contains 569 samples and 30 numerical features used to classify tumors as malignant or benign.
* **Model:** Logistic Regression combined with a `StandardScaler` inside a unified `scikit-learn` Pipeline.

##  Project Structure
```text
ml-production-project/
├── notebooks/
│   └── 01_exploration.ipynb   # Stage A: Exploratory data analysis & baseline validation
├── production_app/            # Core production package
│   ├── __init__.py            # Package marker
│   ├── config.py              # Centralized configuration via environment variables
│   ├── logging_config.py      # Structured logging setup
│   ├── data.py                # Data loading, inspection, and stratified splitting
│   ├── preprocessing.py       # Reusable feature scaling definitions
│   ├── training.py            # Pipeline assembly and model fitting
│   ├── evaluation.py          # Metric reporting and confusion matrix generation
│   ├── model_service.py       # Serialization (joblib) & exception-safe inference
│   └── main.py                # Workflow orchestration
├── models/                    # Saved pipeline artifacts (.joblib)
├── .env.example               # Safe template for environment variables
├── .gitignore                 # Excluded directories (.venv, caches, etc.)
├── .pre-commit-config.yaml    # Code quality and formatting hooks config
├── pyproject.toml             # Project metadata and dependencies (uv)
├── uv.lock                    # Locked dependency versions for reproducibility
├── run.py                     # Main application entry point
└── README.md                  # Project documentation
