from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from production_app.config import TEST_SIZE, RANDOM_STATE
from production_app.logging_config import get_logger

logger = get_logger(__name__)

def load_data():
    """Load the Breast Cancer dataset."""
    return load_breast_cancer(as_frame=True)

def inspect_data(df):
    """Log dataset dimensions and missing values."""
    logger.info(f"Dataset shape: {df.shape} | Missing values: {df.isnull().sum().sum()}")

def split_data(df):
    """Separate features/labels and prepare the train/test split."""
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
