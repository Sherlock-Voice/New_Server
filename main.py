import os
import io
import wave
import pandas as pd
import numpy as np
import librosa
import torch
from fastapi import FastAPI, UploadFile, File
from google.cloud import speech
from app.textrank_model import *
# from KoBERT_Model.KoBERT_model import *

# Google STT API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/user1/sherlockvoice_server/app/keyofgstt.json"

app = FastAPI()

dataset = pd.read_csv('/Users/user1/sherlockvoice_server/app/dataset.csv')

num_mfcc = 100
num_mels = 128
num_chroma = 50

result = {} 

# 합성 음성 판별
def extract_features(X, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
    return np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))

def get_sample_rate(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getframerate()

def get_sample_channels(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getnchannels()

# Google STT API + (KoBERT) + TextRank
def transcribe_audio(content, sample_rate_hertz, sample_channels, filename):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    
    sample_rate_hertz = get_sample_rate(io.BytesIO(content))
    sample_channels = get_sample_channels(io.BytesIO(content))
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz,
        language_code="ko-KR",
        model="default",
        audio_channel_count=sample_channels,
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    response = operation.result(timeout=90)

    transcriptions = {}
    if response.results:
        for res in response.results:
            print("Transcript: {}".format(res.alternatives[0].transcript))
            transcriptions[filename] = res.alternatives[0].transcript
            
            #  kobert 모델을 사용하여 음성판별
            # run(transcriptions)

            # 키워드 및 키워드 문장 출력
            print("Keywords: {}".format(summarize_keywords(transcriptions)))

    else:
        print("Not able to transcribe the audio file")

    result[filename] = transcriptions[filename]

# 엔드포인트
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    X, sample_rate = librosa.load(io.BytesIO(content))
    features = extract_features(X, sample_rate)
    
    closest_match_idx = np.argmin(np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1))
    closest_match_label = dataset.iloc[closest_match_idx, -1]
    total_distance = np.sum(np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1))
    closest_match_prob = 1 - (np.linalg.norm(dataset.iloc[closest_match_idx, :-1] - features) / total_distance)
    closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

    if closest_match_label == 'deepfake':
        print(f"This audio file is fake with {closest_match_prob_percentage} percent probability.")
    else:
        print(f"This audio file is real with {closest_match_prob_percentage} percent probability.")
        transcribe_audio(content, sample_rate, get_sample_channels(io.BytesIO(content)), file.filename)

@app.get("/waiting/{filename}")
async def waiting(filename: str):
    if filename in result:
        return {"status": "ready"}
    else:
        return {"status": "processing"}

@app.get("/result/{filename}")
async def get_result(filename: str):
    if filename in result:
        return {"result": result[filename]}
    else:
        return {"error": "No result available"}

