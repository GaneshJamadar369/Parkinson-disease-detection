import os
import random
import subprocess

def test_random_samples():
    data_path = "../dataset"
    healthy_files = [f for f in os.listdir(os.path.join(data_path, "healthy")) if f.endswith('.wav')]
    parkinson_files = [f for f in os.listdir(os.path.join(data_path, "parkinsons")) if f.endswith('.wav')]
    
    # Pick 3 random from each
    test_files = []
    test_files.extend([("../dataset/healthy/" + f, "Healthy") for f in random.sample(healthy_files, 3)])
    test_files.extend([("../dataset/parkinsons/" + f, "Parkinson's") for f in random.sample(parkinson_files, 3)])
    
    print("=== TESTING RANDOM SAMPLES ===")
    for file_path, expected in test_files:
        print(f"\nTesting: {expected} - {os.path.basename(file_path)}")
        subprocess.run(["python", "predict.py", file_path])

if __name__ == "__main__":
    test_random_samples()