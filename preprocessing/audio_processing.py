import os
import whisper
import ffmpeg
import torch
import json
from pydub import AudioSegment
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
import requests

# Initialize Whisper model (choose "medium" for better accuracy)
whisper_model = whisper.load_model("medium")

ELEVEN_LABS_API_KEY = ""
VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # e.g., "21m00Tcm4TlvDq8ikWAM"

# Initialize Pyannote speaker diarization model
diarization_model = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization")

# Define max duration per segment (in seconds)
MAX_SEGMENT_LENGTH = 300  # 5 minutes

def extract_audio(video_path, output_audio_path="output.wav"):
    """
    Extracts audio from a video file and saves it as a WAV file.

    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio file.

    Returns:
        str: Path to the extracted audio file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")

    # Use FFmpeg to extract audio
    ffmpeg.input(video_path).output(output_audio_path, format='wav', acodec='pcm_s16le', ar='16000').run(overwrite_output=True)
    
    print(f"✅ Audio extracted: {output_audio_path}")
    return output_audio_path

def split_audio(audio_path):
    """
    Splits a long audio file into smaller chunks.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        list: List of paths to split audio files.
    """
    audio = AudioSegment.from_wav(audio_path)
    duration = len(audio) / 1000  # Convert milliseconds to seconds

    if duration <= MAX_SEGMENT_LENGTH:
        return [audio_path]  # No need to split

    chunk_paths = []
    for i, start_time in enumerate(range(0, int(duration), MAX_SEGMENT_LENGTH)):
        chunk = audio[start_time * 1000 : min((start_time + MAX_SEGMENT_LENGTH) * 1000, len(audio))]
        chunk_path = f"{audio_path.replace('.wav', '')}_part{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    print(f"🔄 Audio split into {len(chunk_paths)} parts.")
    return chunk_paths

def transcribe_audio(audio_path):
    """
    Transcribes audio using Whisper STT.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        dict: Transcription results.
    """
    result = whisper_model.transcribe(audio_path)
    return result["segments"]

def diarize_audio(audio_path):
    """
    Performs speaker diarization on an audio file.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        list: Speaker diarization results.
    """
    diarization = diarization_model(audio_path)
    speaker_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })

    return speaker_segments

def process_video(video_path):
    """
    Processes a video file: extracts audio, transcribes speech, and applies speaker diarization.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        dict: Structured transcription and speaker diarization results.
    """
    audio_path = extract_audio(video_path)
    audio_chunks = split_audio(audio_path)

    results = []

    for chunk_path in audio_chunks:
        transcription = transcribe_audio(chunk_path)
        diarization = diarize_audio(chunk_path)

        # Match speaker segments with transcription
        for segment in transcription:
            start, end = segment["start"], segment["end"]
            speaker = "Unknown"

            for diarized in diarization:
                if diarized["start"] <= start and diarized["end"] >= end:
                    speaker = diarized["speaker"]
                    break

            results.append({
                "speaker": speaker,
                "start_time": start,
                "end_time": end,
                "text": segment["text"]
            })

    return results

# Example usage
if __name__ == "__main__":
    video_file = "data/sample_video.mp4"
    output = process_video(video_file)

    # Save results to JSON
    with open("transcription.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n===== FINAL TRANSCRIPTION =====")
    print(json.dumps(output, indent=4))


from google.cloud import speech_v1p1beta1 as speech
import anthropic
# == VIDEO TO TEXT -- NEED GOOGLE CLOUD API
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def extract_info_claude(transcript):
    prompt = f"""
    You are a professional conversationalist, and you are very good at making summaries. 
    You will be given a transcription between speaker 1 and speaker 2. Then, you will be given a set of words, each being annotated
    with whether they belong to speaker 1 or speaker 2. The first to say "my friend" is the user.
    Extract the following information from the speaker that is not face blind. Also, give the timestamp in which the interlocutor 
    gives out their names. 
    Transcript:
    {transcript}
    The 'Other Infos' part should contain at most 5 short bullet points 
    Provide the response in this JSON format:
    {{
        "name": "[Insert the name here]",
        "other_info": "Insert the bullet points here"
        "timestamp : [Insert the timestamp in seconds here]"
    }}
    DO NOT WRITE ANYTHING OUTSIDE OF BRACKETS
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def text_to_speech(json_data):
    prompt = f"""Summarize the following json data using plain sentences
    data : {json_data}. JUST WRITE A SUMMARY, NOTHING ELSE"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.content[0].text
    # The text you want to convert to speech

    # Construct the API endpoint URL
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    # Define the request headers and payload
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": summary,
        "voice_settings": {
            "stability": 0.5,         # Adjust stability as needed (0 to 1)
            "similarity_boost": 0.75    # Adjust similarity boost as needed (0 to 1)
        }
    }

    # Make the POST request to the ElevenLabs API
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the binary response content as an MP3 file
        output_filename = "output.mp3"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"MP3 file saved as {output_filename}")
    else:
        print("Error:", response.status_code, response.text)

