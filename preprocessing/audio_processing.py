import os
import whisper
import ffmpeg
import torch
import json
from pydub import AudioSegment
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
import requests

# Initialize Whisper model (choose "base" for better accuracy on CPU)
whisper_model = whisper.load_model("base")

ELEVEN_LABS_API_KEY = ""
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # e.g., "21m00Tcm4TlvDq8ikWAM"

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
        dict: Transcription results (list of segments).
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

def process_video(video_path, use_diarization=False):
    """
    Processes a video file: extracts audio, transcribes speech, and optionally applies speaker diarization.

    Args:
        video_path (str): Path to the input video file.
        use_diarization (bool): Whether to apply speaker diarization (default True).

    Returns:
        list: Structured transcription (and diarization) results.
    """
    audio_path = extract_audio(video_path)
    audio_chunks = split_audio(audio_path)

    results = []

    for chunk_path in audio_chunks:
        transcription = transcribe_audio(chunk_path)
        if use_diarization:
            diarization = diarize_audio(chunk_path)
        else:
            diarization = []  # No diarization; default speaker will be used

        # Match speaker segments with transcription segments
        for segment in transcription:
            start, end = segment["start"], segment["end"]
            speaker = "Unknown"
            if use_diarization:
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
    
    # Process video with diarization (set use_diarization=False to skip diarization)
    output = process_video(video_file, use_diarization=True)

    # Save results to JSON
    with open("transcription.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n===== FINAL TRANSCRIPTION =====")
    print(json.dumps(output, indent=4))
