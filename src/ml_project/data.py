from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """1. Load data"""
    return load_breast_cancer(as_frame=True)

def inspect_data(df):
    """2. Inspect data"""
    print(f"Dataset shape: {df.shape} | Missing values: {df.isnull().sum().sum()}")

def split_data(df, test_size=0.2, random_state=42):
    """3. Split train/test (بدون أي Preprocessing هنا لمنع Data Leakage)"""
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test