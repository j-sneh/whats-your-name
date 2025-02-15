#!/usr/bin/env python
import cv2
import time
import os
import threading
import numpy as np
import ffmpeg
import sounddevice as sd
import wave
from pydub import AudioSegment
from playsound import playsound

# Assume these functions are defined in your modules:
# face_detection.process_image_embedding(frame) → returns a face embedding or None.
# preprocessing_pipeline.preprocess_video(video_path) → returns dict with keys: "name", "context", "face_embedding", "video_path"
# text_to_speech(text) → generates an audio file from text and returns the file path.
from preprocessing import face_detection, preprocessing_pipeline
from postprocessing.text_to_speech import text_to_speech
from database import InMemoryDatabase

# Initialize the in-memory database
db = InMemoryDatabase()

# Similarity threshold for face matching
SIMILARITY_THRESHOLD = 0.5

# Duration for each video chunk in seconds
CHUNK_DURATION = 15

# Reduced video resolution (width, height) for faster processing
REDUCED_RESOLUTION = (320, 240)

# Audio recording parameters
AUDIO_RATE = 8000       # Lower sampling rate (Hz) for faster processing
AUDIO_CHANNELS = 1      # Mono audio

LAST_KEY = None

def record_audio(duration, output_audio_path):
    """
    Records audio from the default microphone for the given duration and saves as WAV.
    """
    print("🎤 Starting audio recording...")
    audio_data = sd.rec(int(duration * AUDIO_RATE), samplerate=AUDIO_RATE, channels=AUDIO_CHANNELS, dtype='int16')
    sd.wait()
    # Save audio using wave module
    with wave.open(output_audio_path, 'wb') as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio => 2 bytes
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(audio_data.tobytes())
    print(f"✅ Audio recorded: {output_audio_path}")

def merge_audio_video(video_file, audio_file, output_file):
    """
    Merges video and audio files using ffmpeg.
    """
    print("🔄 Merging video and audio...")
    try:
        (
            ffmpeg
            .input(video_file)
            .input(audio_file)
            .output(output_file, vcodec="libx264", acodec="aac", strict="experimental", shortest=None)
            .run(overwrite_output=True)
        )
        print(f"✅ Merged file saved as {output_file}")
    except Exception as e:
        print(f"❌ Error merging audio and video: {e}")

def process_chunks(audio_path, video_path):
    global LAST_KEY
    
    """
    Process a merged video chunk: run the preprocessing pipeline and update the database.
    If a similar face is found, update the summary and generate an audio summary.
    Otherwise, insert a new record.
    """
    info = preprocessing_pipeline.preprocess_video(video_path, audio_path)
    name = info.get("name", "Unknown")
    context = info.get("context", "")
    face_embedding = info.get("face_embedding", None)

    if face_embedding is None:
        print(f"🚨 No face embedding extracted from {chunk_path}.")
        return
    
    match, key = db.extract_data_from_face_embedding(face_embedding, SIMILARITY_THRESHOLD)
    if match is not None and match != LAST_KEY:
        updated_summary = match["summary"] + " " + context
        db.update(face_embedding, match["name"], updated_summary)
        print(f"✅ Updated record for {match['name']}.")
        audio_path = text_to_speech(match["summary"])
        print(f"🔊 Audio summary generated: {audio_path}")
        LAST_KEY = key
        threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()
    else:
        db.insert(face_embedding, name, context)
        print(f"✅ Inserted new record for {name}.")
        audio_path = text_to_speech(context)
        print(f"🔊 Audio summary generated: {audio_path}")
        threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()

def record_and_process_loop():
    # Open the camera (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("🚨 Could not open camera!")
        return

    # Set camera resolution to reduced resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REDUCED_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REDUCED_RESOLUTION[1])

    # Initialize timing and frame storage for a chunk
    start_chunk_time = time.time()
    video_frames = []

    # Start the audio recording thread for the first chunk immediately
    chunk_audio_path = f"temp_audio_{int(start_chunk_time)}.wav"
    audio_thread = threading.Thread(target=record_audio, args=(CHUNK_DURATION, chunk_audio_path))
    audio_thread.start()

    print("✅ Camera activated. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🚨 Frame capture failed, exiting loop.")
            break

        # Resize frame to reduced resolution (if needed)
        frame = cv2.resize(frame, REDUCED_RESOLUTION)
        cv2.imshow("Camera", frame)

        # Optional: perform real-time face detection on a subset of frames (e.g., every 10th frame)
        if int(time.time() * 10) % 10 == 0:
            embedding = face_detection.process_image_embedding(frame)
            if embedding is not None:
                match = db.extract_data_from_face_embedding(embedding, SIMILARITY_THRESHOLD)
                if match is not None:
                    print(f"✅ Real-time: Found match for {match['name']}.")
                    audio_path = text_to_speech(match["summary"])
                    threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()
                else:
                    print("ℹ️ Real-time: Face detected but no match in database.")

        video_frames.append(frame)

        # Check if CHUNK_DURATION has elapsed
        if time.time() - start_chunk_time > CHUNK_DURATION:
            # Ensure the audio recording for this chunk is complete
            audio_thread.join()

            # Save video chunk (using lower quality codec and lower frame rate for speed)
            chunk_video_path = f"temp_chunk_{int(time.time())}.avi"
            height, width, _ = video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Lower frame rate (e.g., 10 FPS) to speed up writing
            out = cv2.VideoWriter(chunk_video_path, fourcc, 10.0, (width, height))
            for f in video_frames:
                out.write(f)
            out.release()
            print(f"🔄 Recorded video chunk saved: {chunk_video_path}")

            # Merge video and audio into one file
            # merged_chunk_path = f"merged_chunk_{int(time.time())}.mp4"
            # merge_audio_video(chunk_video_path, chunk_audio_path, merged_chunk_path)

            # Process the merged video chunk in a separate thread
            threading.Thread(target=process_chunks, args=(chunk_audio_path, chunk_video_path), daemon=True).start()

            # Reset for the next chunk: update timer, clear frames,
            # and start a new audio recording thread concurrently.
            start_chunk_time = time.time()
            video_frames = []
            chunk_audio_path = f"temp_audio_{int(start_chunk_time)}.wav"
            audio_thread = threading.Thread(target=record_audio, args=(CHUNK_DURATION, chunk_audio_path))
            audio_thread.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🚪 Exiting capture loop.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_and_process_loop()
