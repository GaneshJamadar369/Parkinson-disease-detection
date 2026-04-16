import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from features import extract_features

def evaluate_model():
    print("=== MODEL EVALUATION ===")

    # Load data
    X, y = [], []
    data_path = "../dataset"

    for label in ["healthy", "parkinsons"]:
        folder = os.path.join(data_path, label)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                features = extract_features(path)
                X.append(features)
                y.append(0 if label == "healthy" else 1)
            except Exception as e:
                print(f"Error in file: {path} - {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset size: {len(X)} samples")
    print(f"Healthy: {sum(y == 0)}, Parkinson's: {sum(y == 1)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model
    model = pickle.load(open("parkinson_model.pkl", "rb"))

    # Train on full training set (simulate training)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for AUC
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n--- TRAINING SET ---")
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Training AUC: {roc_auc_score(y_train, y_train_prob):.4f}")

    print("\n--- TEST SET ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test AUC: {roc_auc_score(y_test, y_test_prob):.4f}")

    print("\n--- CONFUSION MATRIX (Test Set) ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    print("\n--- CLASSIFICATION REPORT (Test Set) ---")
    print(classification_report(y_test, y_test_pred, target_names=['Healthy', 'Parkinson\'s']))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_test_prob):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Parkinson\'s Detection')
    plt.legend()
    plt.savefig('roc_curve.png')
    print("\n✅ ROC curve saved as 'roc_curve.png'")

if __name__ == "__main__":
    evaluate_model()