import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from features import extract_features

def test_on_dataset():
    # 📦 Load model
    model = pickle.load(open("parkinson_model.pkl", "rb"))
    
    X_test, y_test = [], []
    data_path = "../dataset"
    
    for label in ["healthy", "parkinsons"]:
        folder = os.path.join(data_path, label)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                features = extract_features(path)
                X_test.append(features)
                y_test.append(0 if label == "healthy" else 1)
            except:
                print(f"Error in file: {path}")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # 🔍 Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For AUC
    
    # 📊 Metrics
    print("=== MODEL EVALUATION ON FULL DATASET ===")
    print(f"Total samples: {len(y_test)}")
    print(f"Healthy: {sum(y_test == 0)}, Parkinson's: {sum(y_test == 1)}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Accuracy per class
    healthy_acc = cm[0,0] / cm[0,:].sum()
    parkinson_acc = cm[1,1] / cm[1,:].sum()
    print(f"\nHealthy accuracy: {healthy_acc:.4f}")
    print(f"Parkinson's accuracy: {parkinson_acc:.4f}")

if __name__ == "__main__":
    test_on_dataset()