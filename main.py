import os
import io
import wave
import pandas as pd
import numpy as np
import librosa
import torch
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import speech
from app.textrank_model import *
from multiprocessing import Process
# from KoBERT_Model.KoBERT_model import *
from typing import Optional
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Google STT API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/user1/sherlockvoice_server/app/sherlock-voice-c074041bd62d.json"

app = FastAPI()

# CORS 설정
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = False,  # cookie 설정
    allow_methods=["*"],     
    allow_headers=["*"],     
)

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

    return summarize_keywords(transcriptions)

# 유효성 검사 오류 처리기
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

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

    # Hash the file name and convert it to an integer within a certain range
    task_id = int(hashlib.sha256(file.filename.encode()).hexdigest(), 16) % 1000000  # Adjust the range as needed
    result[task_id] = {}
    
    # Initialize the dictionary for the current file
    if closest_match_label == 'deepfake':
        print(f"This audio file is fake with {closest_match_prob_percentage} percent probability.")
        result[task_id] = {"closest_match_prob_percentage": closest_match_prob_percentage}
    else:
        print(f"This audio file is real with {closest_match_prob_percentage} percent probability.")
        result[task_id] = {
            "closest_match_prob_percentage": closest_match_prob_percentage,
            "keywords": transcribe_audio(content, sample_rate, get_sample_channels(io.BytesIO(content)), file.filename)
        }
        
        # Start transcribing audio in a separate process
        p = Process(target=transcribe_audio, args=(content, sample_rate, get_sample_channels(io.BytesIO(content)), file.filename))
        p.start()
        
    return {"task_id": task_id}
        
@app.get("/waiting/{task_id}")
async def waiting(task_id: int):
    # Check if the task is completed
    if task_id in result:
        return {"status": "ready"}
    else:
        return {"status": "processing"}

@app.get("/result/{task_id}")
async def get_result(task_id: int):
    if task_id in result:
        if len(result[task_id]) == 1:
            return {"closest_match_prob_percentage": result[task_id]["closest_match_prob_percentage"]}
        else:
            return {"closest_match_prob_percentage": result[task_id]["closest_match_prob_percentage"], "keywords": result[task_id]["keywords"]}
    else:
        return {"error": "No result available or closest_match_prob_percentage not calculated yet"}


# uvicorn app.main:app --reload  