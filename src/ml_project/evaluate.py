from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, target_names):
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return acc