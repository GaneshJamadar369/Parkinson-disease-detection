import sounddevice as sd
from scipy.io.wavfile import write
import pickle
import numpy as np
import time
import sys
import os
from features import extract_features

def predict_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📂 Loading audio from file: {file_path}")
    
    # 📦 Load model
    model = pickle.load(open("parkinson_model.pkl", "rb"))
    
    # 🧠 Feature extraction
    features = extract_features(file_path).reshape(1, -1)
    print("Feature length:", len(features[0]))
    
    # 🔍 Prediction
    prediction = model.predict(features)
    
    # 📊 Result
    print("\n🔎 RESULT:")
    if prediction[0] == 1:
        print("⚠️ Parkinson's Detected")
    else:
        print("✅ Healthy Person")

def record_and_predict():
    # 📢 Sentence for user
    sentence = "Parkinson's disease detection test"
    
    print("\n==============================")
    print("📢 Please read the sentence clearly:")
    print(f"\n👉 {sentence}")
    print("\nRecording will start in 3 seconds...")
    print("==============================\n")
    
    time.sleep(3)
    
    # 🎤 Record audio
    fs = 44100
    seconds = 5
    
    print("🎙️ Recording... Speak now!")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    
    write("test.wav", fs, recording)
    
    print("✅ Recording complete!\n")
    
    predict_from_file("test.wav")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        predict_from_file(file_path)
    else:
        record_and_predict()