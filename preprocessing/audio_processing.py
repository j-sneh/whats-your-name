# import os
# import subprocess
# import cv2
# import face_recognition
# import sqlite3
# import pyttsx3
# import ffmpeg  # requires ffmpeg-python
# import wave
from google.cloud import speech #, aiplatform, storage
# import contextlib
# import anthropic
import json

# == VIDEO TO TEXT -- NEED GOOGLE CLOUD API


with open("secrets.json", "r", encoding="utf-8") as file:
    secrets = json.load(file)
    GOOGLE_CLOUD_API_KEY = secrets["google_cloud_api_key"]
    ANTHROPIC_API_KEY = secrets["anthropic_api_key"]

def transcribe_speech(speech_file) -> str:
    # If you're using ADC, no need to pass client_options
    client = speech.SpeechClient(client_options={"api_key": GOOGLE_CLOUD_API_KEY })

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

    
    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    
    response = client.recognize(config=config, audio=audio)
    print(response)
    
    transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    
    return transcript

if __name__ == "__main__":
    print(transcribe_speech("Hello.mp3"))