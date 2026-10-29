from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from production_app.config import RANDOM_STATE, TEST_SIZE
from production_app.logging_config import get_logger

logger = get_logger(__name__)

# A list of the 5 best features selected
SELECTED_FEATURES = [
    "mean_perimeter",
    "mean_concave_points",
    "worst_radius",
    "worst_perimeter",
    "worst_concave_points",
]


def load_data():
    """Load the Breast Cancer dataset and clean column names."""
    data = load_breast_cancer(as_frame=True)

    # Convert spaces to underscores to match the schemas.py file.
    data.frame.columns = data.frame.columns.str.replace(" ", "_")
    data.feature_names = [name.replace(" ", "_") for name in data.feature_names]

    return data


def inspect_data(df):
    """Log dataset dimensions and missing values."""
    logger.info(
        f"Dataset shape: {df.shape} | Missing values: {df.isnull().sum().sum()}"
    )


def split_data(df):
    """Separate features/labels and prepare the train/test split."""
    # Passing only the 5 features to the model instead of the 30
    X = df[SELECTED_FEATURES]
    y = df["target"]

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
