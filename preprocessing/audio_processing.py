import os
import subprocess
import cv2
import face_recognition
import sqlite3
import pyttsx3
import ffmpeg  # requires ffmpeg-python
import wave
from google.cloud import aiplatform, storage, speech
import contextlib
import anthropic

# == VIDEO TO TEXT -- NEED GOOGLE CLOUD API

def transcribe_speech(speech_uri) -> str:
    # If you're using ADC, no need to pass client_options
    client = speech.SpeechClient()

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="phone_call",
        diarization_config=diarization_config,
    )

    
    audio = speech.RecognitionAudio(uri=speech_uri)
    
    response = client.recognize(config=config, audio=audio)
    
    transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    
    return transcript