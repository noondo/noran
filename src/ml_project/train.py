from ml_project.data import load_data, inspect_data, split_data
from ml_project.model import build_and_fit_pipeline, save_pipeline
from ml_project.evaluate import evaluate_model

def main():
    print("Step 1: Loading data...")
    raw_data = load_data()
    df = raw_data.frame
    
    print("Step 2: Inspecting data...")
    inspect_data(df)
    
    print("Step 3: Splitting train/test...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Step 4 & 5: Preprocessing and fitting Pipeline (No Data Leakage)...")
    pipeline = build_and_fit_pipeline(X_train, y_train)
    
    print("Step 6: Evaluating Pipeline...\n")
    evaluate_model(pipeline, X_test, y_test, raw_data.target_names)
    
    print("Step 7: Saving fitted pipeline to 'models/pipeline.joblib'...")
    save_pipeline(pipeline, raw_data.target_names)
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()