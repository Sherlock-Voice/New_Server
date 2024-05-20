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
from textrank_model import *
from deepfake_model import *
from multiprocessing import Process
# from KoBERT_Model.KoBERT_model import *
from typing import Optional
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tempfile

# Google STT API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/user1/SherlockVoice/sherlockvoice_server/app/sherlock-voice-da3e0129f362.json"

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

result = {} 

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

    # Create a temporary file and write the content to it
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(content)
    temp_file.close()
    
    file_name = temp_file.name  # Use the temporary file's name as the file name
    
    # Hash the file name and convert it to an integer within a certain range
    task_id = int(hashlib.sha256(file.filename.encode()).hexdigest(), 16) % 1000000  # Adjust the range as needed
    result = {}
    
    deepfake_result = run_audio_classifier(file_name)
    
    if 'fake' in deepfake_result[file_name]['result']:
        print(f"This audio file is fake with {deepfake_result[file_name]['prob']} percent probability.")
        result[task_id] = {"closest_match_prob_percentage": deepfake_result[file_name]['prob']}
    else:
        print(f"This audio file is real with {deepfake_result[file_name]['prob']} percent probability.")
        result[task_id] = {
            "closest_match_prob_percentage": deepfake_result[file_name]['prob'],
            "keywords": transcribe_audio(content, get_sample_rate(io.BytesIO(content)), get_sample_channels(io.BytesIO(content)), file.filename)
        }
        # Start transcribing audio in a separate process
        p = Process(target=transcribe_audio, args=(content, get_sample_rate(io.BytesIO(content)), get_sample_channels(io.BytesIO(content)), file.filename))
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