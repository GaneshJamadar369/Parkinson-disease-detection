import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from features import extract_features   # ✅ IMPORTANT

# ---------------- LOAD DATA ----------------
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
        except:
            print(f"Error in file: {path}")

X = np.array(X)
y = np.array(y)

# ---------------- DATA CHECK ----------------
print("Data distribution:", Counter(y))
print("Feature length:", len(X[0]))   # ✅ correct place

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ---------------- MODEL ----------------
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced', probability=True)
)

# ---------------- TRAIN ----------------
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- SAVE ----------------
pickle.dump(model, open("parkinson_model.pkl", "wb"))

print("\n✅ Model Saved Successfully!")