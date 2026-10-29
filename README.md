## Breast Cancer Classification — Production ML Pipeline

## 1. Project Goal
The objective of this project is to transition an experimental machine learning workflow from a Jupyter Notebook into a modular, production-ready Python application. The pipeline predicts whether a breast tumor is **malignant** or **benign** based on diagnostic features, adhering to software engineering best practices including environment-based configuration, structured logging, exception handling, artifact serialization, and automated code-quality checks (`Ruff`, `Black`, and `pre-commit`).

## 2. Dataset
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (`scikit-learn` built-in dataset).
- **Shape:** 569 samples and 31 columns (30 numeric predictive features + 1 target column).
- **Missing Values:** 0 missing values.
- **Target Classes:** `malignant` (0) and `benign` (1).

## 3. ML Pipeline & Model Choice
To prevent data leakage and ensure seamless reproducibility between training and inference, the workflow wraps preprocessing and classification into a single `scikit-learn` `Pipeline`:
1. **Preprocessing (`StandardScaler`):** Standardizes numerical features by removing the mean and scaling to unit variance.
2. **Model (`LogisticRegression`):** Selected for its strong baseline performance on linearly separable medical tabular data, fast training speed, and high interpretability.
3. **Serialization (`joblib`):** Both the fitted preprocessor and trained model are persisted together inside `models/pipeline.joblib` alongside target class names (`models/target_names.joblib`). Prediction loads this exact artifact without retraining.

## 4. Folder Structure
```text
ml-project/
├── models/                     # Saved serialized pipeline and target names (.joblib)
├── notebooks/
│   └── 01_exploration.ipynb    # Initial EDA, experimentation, and baseline modeling
├── production_app/             # Modular production Python package
│   ├── __init__.py
│   ├── config.py               # Environment variable loading and path configurations
│   ├── data.py                 # Dataset loading, inspection, and train/test splitting
│   ├── evaluation.py           # Model evaluation metrics, confusion matrix, and reports
│   ├── logging_config.py       # Centralized logging configuration
│   ├── main.py                 # Core orchestration for training and inference workflows
│   ├── model_service.py        # Artifact saving, loading, and single-sample inference
│   ├── preprocessing.py        # Feature scaling and preprocessing pipeline builder
│   └── training.py             # Pipeline assembly and model fitting
├── .env.example                # Safe template for runtime environment variables
├── .gitignore                  # Ignored files (.env, __pycache__, virtual environments)
├── .pre-commit-config.yaml     # Automated pre-commit hooks for Ruff and Black
├── pyproject.toml              # Project dependencies and tool configurations
├── README.md                   # Project documentation and execution guide
└── run.py                      # CLI entry point for training and prediction

# Production ML Classification Project

## Project Structure
```text
production_app/
├── __init__.py
├── config.py
├── model_service.py
# Assignment 02: Production API Integration
## Assignment 02: Production API Integration

### Overview
This assignment extends the ML pipeline from Assignment 01 by wrapping the trained classification model in a production-ready REST API using FastAPI. The application flow starts with an HTTP request containing patient data, validates the input using Pydantic, loads the saved model artifact, and returns a real-time prediction.

### Installation and Execution
**1. Environment Setup & Dependencies**
Ensure you have `uv` installed, then synchronize the environment:
```bash
uv sync
