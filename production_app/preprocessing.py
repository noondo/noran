from sklearn.preprocessing import StandardScaler

def create_preprocessor():
    """Define reusable preprocessing step (fitted only on training data via Pipeline)."""
    return StandardScaler()
