import librosa
import numpy as np

def add_noise(audio):
    noise = np.random.randn(len(audio))
    return audio + 0.005 * noise

def extract_features(file):
    audio, sr = librosa.load(file, sr=None)
    audio = add_noise(audio)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)

    features = np.hstack([mfcc, spectral, zcr])
    return features