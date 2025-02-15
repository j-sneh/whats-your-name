#!/usr/bin/env python
import cv2
import face_recognition
import sqlite3
import pyttsx3
import speech_recognition as sr
import ffmpeg  # requires ffmpeg-python
import database
import wave
from google.cloud import aiplatform, storage, speech
import contextlib
import anthropic
PATH = ""
# ------------------------------
# CONFIGURATION & FILE PATHS
# ------------------------------
VIDEO_FILE = "input_video.mp4"       # Input video file (recorded using a smartphone)
AUDIO_FILE = "extracted_audio.wav"   # Output audio file after extraction
DB_FILE = "contacts.db"              # SQLite database file
import time
import os
import threading
import numpy as np
from playsound import playsound

# Assume these are implemented in your modules:
# - face_detection(image): returns a face embedding (numpy array) or None if no face is detected.
# - preprocessing_pipeline(video_path): processes a video chunk and returns a dict with keys:
#         "name", "context", "face_embedding", "video_path"
# - text_to_speech(text): generates an audio file from text and returns its file path.
from preprocessing import face_detection, preprocessing_pipeline
from postprocessing import text_to_speech

# In-memory database from your previous implementation
from database import InMemoryDatabase  # Make sure your InMemoryDatabase has a search_by_embedding(threshold, embedding) method.

# Initialize the database (or load an existing one)
db = InMemoryDatabase()

# Similarity threshold (Euclidean distance threshold) for face matching
SIMILARITY_THRESHOLD = 0.5

# Video chunk duration in seconds
CHUNK_DURATION = 15

def process_video_chunk(chunk_path):
    """
    Process a video chunk: run the preprocessing pipeline and update the database.
    If a similar face is found, update the summary and generate an audio summary.
    Otherwise, insert a new record.
    """
    info = preprocessing_pipeline(chunk_path)
    name = info.get("name", "Unknown")
    context = info.get("context", "")
    face_embedding = info.get("face_embedding", None)

    if face_embedding is None:
        print(f"🚨 No face embedding extracted from {chunk_path}.")
        return

    # Search for a matching face in the database
    match = db.search_by_embedding(SIMILARITY_THRESHOLD, face_embedding)
    if match is not None:
        updated_summary = match["summary"] + " " + context
        db.update(face_embedding, match["name"], updated_summary)
        print(f"✅ Updated record for {match['name']}.")
        # Generate TTS from the updated summary and play it
        audio_path = text_to_speech(match["summary"])
        print(f"🔊 Audio summary generated: {audio_path}")
        threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()
    else:
        db.insert(face_embedding, name, context)
        print(f"✅ Inserted new record for {name}.")
        audio_path = text_to_speech(context)
        print(f"🔊 Audio summary generated: {audio_path}")
        threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()

def record_and_process_loop():
    # Open the camera (0 is the default device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("🚨 Could not open camera!")
        return

    # Initialize variables for video chunking
    start_chunk_time = time.time()
    frames = []

    print("✅ Camera activated. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🚨 Frame capture failed, exiting loop.")
            break

        # Optionally display the frame
        cv2.imshow("Camera", frame)

        # Real-time face detection on the current frame
        embedding = face_detection(frame)  # Should return a numpy array or None.
        if embedding is not None:
            # Check database for a similar face
            match = db.extract_data_from_face_embedding(embedding, SIMILARITY_THRESHOLD)
            if match is not None:
                print(f"✅ Real-time: Found match for {match['name']}.")
                # Generate audio summary (if not already playing) in a separate thread
                audio_path = text_to_speech(match)
                threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()
            else:
                print("ℹ️ Real-time: Face detected but no match in database.")
                # You might want to record this instance later when processing video chunks.
        
        # Append current frame to the current chunk
        frames.append(frame)

        # Check if we have recorded enough frames for a CHUNK_DURATION video
        if time.time() - start_chunk_time > CHUNK_DURATION:
            # Save the chunk to a temporary video file
            chunk_file = f"temp_chunk_{int(time.time())}.avi"
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(chunk_file, fourcc, 20.0, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            print(f"🔄 Recorded video chunk saved: {chunk_file}")

            # Process the video chunk in a separate thread
            threading.Thread(target=process_video_chunk, args=(chunk_file,), daemon=True).start()

            # Reset for next chunk
            frames = []
            start_chunk_time = time.time()

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🚪 Exiting capture loop.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_and_process_loop()
