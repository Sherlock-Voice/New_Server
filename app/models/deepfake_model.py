import wave
import numpy as np
import sqlite3
import re
import pandas as pd
import librosa
import os
import time
from scipy.spatial.distance import cosine

data_file = "/Users/user1/SherlockVoice/New_Server/app/models/shuffle_400.csv"


def load_audio_features(file_name, num_mfcc=100, num_mels=128, num_chroma=50):
    X, sample_rate = librosa.load(file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
    return np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))

def find_closest_match(features, shuffle_data):
    similarities = []
    for sample_features in shuffle_data.iloc[:, :-1].values:
        similarity = 1 - cosine(sample_features, features)
        similarities.append(similarity)
    closest_match_idx = np.argmax(similarities)
    closest_match_label = shuffle_data.iloc[closest_match_idx, -1]
    closest_match_similarity = similarities[closest_match_idx]
    return closest_match_label, closest_match_similarity

def classify_audio(file_name, shuffle_data):
    features = load_audio_features(file_name)
    closest_match_label, closest_match_similarity = find_closest_match(features, shuffle_data)

    closest_match_prob = closest_match_similarity
    closest_match_prob_percentage = "{:.0f}".format(closest_match_prob * 100)
    
    deepfake_result = {}
    
    if closest_match_label == 'fake':
        deepfake_result[file_name] = {'result': 'deepfake', 'prob': closest_match_prob_percentage}
        
    else:
        deepfake_result[file_name] = {'result': 'real', 'prob': closest_match_prob_percentage}
    return deepfake_result

def run_audio_classifier(file_name):
    shuffle_data = pd.read_csv(data_file)
    deepfake_result = classify_audio(file_name, shuffle_data)
    print(deepfake_result)
    return deepfake_result



# if __name__ == "__main__":
#     file_name = "/Users/user1/Desktop/Sample_audio/sample_001.wav"  # 입력받은 오디오파일명 입력
#     run_audio_classifier(file_name)