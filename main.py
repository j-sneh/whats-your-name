import os
import subprocess
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

# ------------------------------
# VIDEO AND AUDIO EXTRACTION
# ------------------------------
def extract_audio(video_path, audio_path):
    """
    Extract audio from video using ffmpeg.
    """
    # Command: extract audio to wav format (pcm_s16le) at 44100 Hz, stereo.
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_path,
        "-y"  # Overwrite if exists
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Audio extracted to {audio_path}")

def extract_frames(video_path, frame_interval=30):
    """
    Extract frames from video using OpenCV.
    Returns a list of (frame_number, frame image) tuples.
    frame_interval defines how many frames to skip between extractions.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from video.")
    return frames

# ------------------------------
# VOICE ACTIVITY & SPEECH-TO-TEXT
# ------------------------------
def speech_to_text(audio_path):
    """
    Convert the entire audio file to text using SpeechRecognition.
    For MVP purposes, we process the entire audio.
    """
    recognizer = sr.Recognizer()
    transcript = ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Error during speech recognition: {e}")
    print("Transcript:", transcript)
    return transcript

# ------------------------------
# FACE DETECTION AND RECOGNITION
# ------------------------------
def detect_faces_in_frame(frame):
    """
    Detect faces in a given frame using face_recognition.
    Returns a list of face encodings and their locations.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_locations, face_encodings

def process_frames_for_face(frames):
    """
    Process extracted frames and return the first detected face encoding along with the corresponding frame.
    (For simplicity, we only process and store one face.)
    """
    for (num, frame) in frames:
        locs, encodings = detect_faces_in_frame(frame)
        if encodings:
            print(f"Face detected in frame {num}")
            return encodings[0], frame  # Return first face encoding and frame
    print("No face detected in any frame.")
    return None, None

# ------------------------------
# DATABASE FUNCTIONS
# ------------------------------
def init_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    info TEXT,
     -- Assuming 512-dim embeddings
            );
    ''')
    conn.commit()
    conn.close()

def store_contact(db_file, name, encoding, transcript):
    """
    Store a contact's data into the SQLite database.
    We store the face encoding as a comma-separated string.
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Convert the encoding (a numpy array) to a string for simplicity
    encoding_str = ",".join([str(x) for x in encoding])
    c.execute('''
        INSERT INTO contacts (name, transcript)
        VALUES (?, ?, ?)
    ''', (name, encoding_str, transcript))
    conn.commit()
    conn.close()

def retrieve_contacts(db_file, id):
    """
    Retrieve all stored contacts from the database.
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT * FROM contacts where id =', id)
    contacts = c.fetchall()
    conn.close()
    return contacts

# ------------------------------
# TEXT-TO-SPEECH
# ------------------------------
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ------------------------------
# MAIN WORKFLOW
# ------------------------------
def main():
    # Step 0: Initialize Database
    init_database(DB_FILE)
    
    # Step 1: Extract Audio from Video
    if not os.path.exists(VIDEO_FILE):
        print(f"Video file {VIDEO_FILE} not found.")
        return
    extract_audio(VIDEO_FILE, AUDIO_FILE)
    
    # Step 2: Extract Frames from Video
    frames = extract_frames(VIDEO_FILE, frame_interval=30)
    
    # Step 3: Process Audio (Speech-to-Text)
    transcript = speech_to_text(AUDIO_FILE)
    
    # Step 4: Process Frames for Face Detection
    face_encoding, face_frame = process_frames_for_face(frames)

    
    if face_encoding is None:
        print("No face found. Exiting.")
        return
    
    key = database.extract_data_from_face_embedding()
    # For the purpose of MVP, we will use a dummy name for the contact.
    name = "Contact 1"
    
    # Step 5: Store Data in Database
    store_contact(DB_FILE, name, face_encoding, transcript)
    
    # Step 6: Simulate Retrieval & TTS Feedback
    contacts = retrieve_contacts(DB_FILE)
    if contacts:
        # For MVP, just take the first contact and read back the stored transcript.
        contact = contacts[0]
        retrieved_name = contact[1]
        retrieved_transcript = contact[3]
        tts_message = f"{retrieved_name}: {retrieved_transcript}"
        print("TTS Message:", tts_message)
        speak_text(tts_message)
    else:
        print("No contacts found in database.")

if __name__ == '__main__':
    main()


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


client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



def extract_info_claude(transcript):
    prompt = f"""
    You are a professional conversationalist, and you are very good at making summaries. 
    You will be given a transcription between speaker 1 and speaker 2. The first to say they are face-blind is the user.
    Extract the following information from the speaker that is not face blind. Also, give the timestamp in which the interlocutor 
    gives out their names
    
    - Name:
    - Other Infos:

    Transcript:
    {transcript}
    The 'Other Infos' part should contain at most 5 short bullet points 
    Provide the response in this JSON format:
    {{
        "name": "Extracted Name",
        "other_info": "Extracted information"
    }}
    """

    response = client.messages.create(
        model="claude-2",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content



def overlay_bbox_and_text(video_path, output_path, bboxes, stamp, stamp_text, text):
    """
    Overlays bounding boxes (with center coordinates) and text on a video, then returns the new video file.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video.
        bboxes (list of tuples): A list of tuples (x, y, w, h) for each frame (starting at stamp seconds),
                                 where (x, y) is the center of the box.
        stamp (float): The time (in seconds) when the bounding boxes start appearing.
        stamp_text (float): The time (in seconds) when the text starts appearing.
        text (str): The text to overlay on the video.
    
    Returns:
        str: The output video file path.
    """
    fps = 30  # Video frame rate
    box_start_frame = int(stamp * fps)
    text_start_frame = int(stamp_text * fps)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # If we're at or after the stamp for bounding boxes...
        if frame_idx >= box_start_frame:
            # Compute relative frame index since stamp.
            rel_frame = frame_idx - box_start_frame
            # If there's a bounding box for this frame...
            if rel_frame < len(bboxes):
                center_x, center_y, w, h = bboxes[rel_frame]
                # Convert center coordinates to top-left corner coordinates.
                top_left_x = int(center_x - w / 2)
                top_left_y = int(center_y - h / 2)
                bottom_right_x = int(center_x + w / 2)
                bottom_right_y = int(center_y + h / 2)
                
                # Draw the bounding box (in green).
                cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 0), 2)
                
                # Overlay text if we're past the stamp for text.
                if frame_idx >= text_start_frame:
                    # Position the text above the bounding box.
                    cv2.putText(frame, text, (top_left_x, top_left_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Output video saved to: {output_path}")
    
    return output_path
