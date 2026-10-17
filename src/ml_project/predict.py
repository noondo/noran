from ml_project.data import load_data
from ml_project.model import load_pipeline

def main():
    print("Step 8: Reloading saved pipeline...")
    pipeline, target_names = load_pipeline()
   
    raw_data = load_data()
    df = raw_data.frame
    sample_features = df.drop(columns=['target']).iloc[[0]]
    actual_label = df['target'].iloc[0]
    
    print("Step 9: Predicting with pipeline (automatic preprocessing)...")
    pred_class = pipeline.predict(sample_features)[0]
    pred_prob = pipeline.predict_proba(sample_features)[0]
    
    print("\n=== Prediction Result ===")
    print(f"Actual Label    : {target_names[actual_label]} ({actual_label})")
    print(f"Predicted Label : {target_names[pred_class]} ({pred_class})")
    print(f"Confidence      : {pred_prob[pred_class]*100:.2f}%")

if __name__ == "__main__":
    main()