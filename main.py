#!/usr/bin/env python
import cv2
import time
import threading
import numpy as np
import ffmpeg
import sounddevice as sd
import wave
from pydub import AudioSegment
from playsound import playsound

# Assume these functions are defined in your modules:
# face_detection.process_image_embedding(frame) → returns a face embedding or None.
# preprocessing_pipeline.preprocess_video(video_path, audio_path) → returns dict with keys: "name", "context", "face_embedding", "video_path"
# text_to_speech(text) → generates an audio file from text and returns the file path.
from preprocessing import face_detection, preprocessing_pipeline
from postprocessing.text_to_speech import text_to_speech
from database import InMemoryDatabase

# Initialize the in-memory database
db = InMemoryDatabase()

# Similarity threshold for face matching
SIMILARITY_THRESHOLD = 0.5

# Duration for each video/audio chunk in seconds
CHUNK_DURATION = 15

# Reduced video resolution (width, height) for faster processing
REDUCED_RESOLUTION = (320, 240)

# Audio recording parameters
AUDIO_RATE = 8000       # Lower sampling rate (Hz)
AUDIO_CHANNELS = 1      # Mono audio

# Global variable to coordinate duplicate audio summary generation
LAST_KEY = None

# --- Worker functions ---

def record_audio(duration, output_audio_path):
    """
    Records audio from the default microphone for the given duration and saves it as a WAV file.
    """
    print("🎤 [Audio] Starting audio recording...")
    audio_data = sd.rec(int(duration * AUDIO_RATE), samplerate=AUDIO_RATE, channels=AUDIO_CHANNELS, dtype='int16')
    sd.wait()
    with wave.open(output_audio_path, 'wb') as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio => 2 bytes
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(audio_data.tobytes())
    print(f"✅ [Audio] Audio recorded: {output_audio_path}")

def audio_capture_worker(duration, output_audio_path):
    """
    Worker function for audio capture.
    """
    record_audio(duration, output_audio_path)

def video_capture_worker(stop_event, video_frames):
    """
    Worker function for video capture. Opens the camera, captures frames until stop_event is set,
    shows the frames (and performs real-time face detection), and appends them to video_frames list.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("🚨 [Video] Could not open camera!")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REDUCED_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REDUCED_RESOLUTION[1])
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("🚨 [Video] Frame capture failed.")
            break

        # Resize frame (if needed)
        frame = cv2.resize(frame, REDUCED_RESOLUTION)
        video_frames.append(frame)

        # Optional: real-time face detection (using your module)
        # This block uses your face detection logic and plays audio only if a new face is detected.
        embedding = face_detection.process_image_embedding(frame)
        if embedding is not None:
            result = db.extract_data_from_face_embedding(embedding, SIMILARITY_THRESHOLD)
            if result is not None:
                match, key = result
                # Only play audio if this is a new face (i.e. key is different)
                global LAST_KEY
                if key != LAST_KEY:
                    LAST_KEY = key
                    print(f"✅ [Real-time] New match for {match['name']}.")
                    audio_out = text_to_speech(match["summary"])
                    threading.Thread(target=playsound, args=(audio_out,), daemon=True).start()
                else:
                    print(f"⏳ [Real-time] Duplicate match ({match['name']}); skipping.")
            else:
                print("ℹ️ [Real-time] Face detected but no match in database.")

        # Show the frame (for debugging)
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ [Video] Video capture thread ending.")

def process_chunks(audio_path, video_path):
    """
    Processes the video and audio chunk:
      - Runs the preprocessing pipeline.
      - Updates/inserts records in the database.
      - Generates and plays an audio summary if a new face is detected.
    """
    global LAST_KEY
    info = preprocessing_pipeline.preprocess_video(video_path, audio_path)
    name = info.get("name", "Unknown")
    context = info.get("context", "")
    face_embedding = info.get("face_embedding", None)

    if face_embedding is None:
        print("🚨 [Process] No face embedding extracted.")
        return

    result = db.extract_data_from_face_embedding(face_embedding, SIMILARITY_THRESHOLD)
    if result is not None:
        match, key = result
        # Only update/generate audio if the new key is different from LAST_KEY
        if key != LAST_KEY:
            LAST_KEY = key
            updated_summary = match["summary"] + " " + context
            db.update(face_embedding, match["name"], updated_summary)
            print(f"✅ [Process] Updated record for {match['name']}.")
            audio_out = text_to_speech(match["summary"])
            print(f"🔊 [Process] Audio summary generated: {audio_out}")
            threading.Thread(target=playsound, args=(audio_out,), daemon=True).start()
        else:
            print(f"⏳ [Process] Duplicate match ({match['name']}) detected; skipping audio summary.")
    else:
        db.insert(face_embedding, name, context)
        print(f"✅ [Process] Inserted new record for {name}.")
        audio_out = text_to_speech(context)
        print(f"🔊 [Process] Audio summary generated: {audio_out}")
        threading.Thread(target=playsound, args=(audio_out,), daemon=True).start()

# --- Main loop that coordinates threads ---

def record_and_process_loop():
    print("✅ Starting concurrent audio and video capture. Press 'q' in the video window to quit.")
    
    while True:
        # Containers for the current chunk
        video_frames = []
        chunk_start_time = time.time()
        # Prepare audio file path for this chunk
        chunk_audio_path = f"temp_audio_{int(chunk_start_time)}.wav"
        
        # Create an Event to signal the video thread to stop after CHUNK_DURATION.
        stop_event = threading.Event()
        
        # Start the video capture thread (it appends frames to video_frames)
        video_thread = threading.Thread(target=video_capture_worker, args=(stop_event, video_frames))
        video_thread.start()
        
        # Start the audio capture thread concurrently
        audio_thread = threading.Thread(target=audio_capture_worker, args=(CHUNK_DURATION, chunk_audio_path))
        audio_thread.start()
        
        # Let both threads run for CHUNK_DURATION seconds
        time.sleep(CHUNK_DURATION)
        # Signal video thread to stop capturing
        stop_event.set()
        
        # Wait for both threads to finish
        video_thread.join()
        audio_thread.join()
        
        # Save the captured video frames to a video file
        if not video_frames:
            print("🚨 No video frames captured; skipping processing for this chunk.")
            continue
        
        chunk_video_path = f"temp_chunk_{int(time.time())}.avi"
        height, width, _ = video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(chunk_video_path, fourcc, 10.0, (width, height))
        for frame in video_frames:
            out.write(frame)
        out.release()
        print(f"🔄 [Main] Recorded video chunk saved: {chunk_video_path}")
        
        # Process the audio and video chunk in a separate thread
        threading.Thread(target=process_chunks, args=(chunk_audio_path, chunk_video_path), daemon=True).start()
        
        # Optionally, check if the user pressed 'q' in the video window.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🚪 [Main] Exiting capture loop.")
            break

if __name__ == "__main__":
    record_and_process_loop()
