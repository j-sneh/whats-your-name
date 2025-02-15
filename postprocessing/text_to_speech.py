#!/usr/bin/env python
import os
import sys
import time
from TTS.api import TTS

# Threshold for warning about text length (e.g., 500 characters)
TEXT_LENGTH_THRESHOLD = 500

def text_to_speech(text, output_folder="output_audio", threshold=TEXT_LENGTH_THRESHOLD):
    """
    Converts input text to speech using Coqui TTS, saves the audio file, and returns the output folder path.

    Args:
        text (str): The text to synthesize.
        output_folder (str): Folder where the audio file will be saved.
        threshold (int): Maximum recommended text length before printing a warning.

    Returns:
        str: The path to the output folder.
    """
    # Warn if the text is too long
    if len(text) > threshold:
        print("⚠️ Warning: The input text is quite long. Synthesis may take longer than usual.")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate a unique file name using the current timestamp
    timestamp = int(time.time())
    file_name = f"tts_output_{timestamp}.wav"
    output_path = os.path.join(output_folder, file_name)
    
    # Initialize the Coqui TTS model (using a pre-trained model, CPU-only)
    print("🔄 Initializing Coqui TTS model...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    
    # Synthesize the text and save to file
    print("🔄 Converting text to speech...")
    tts.tts_to_file(text=text, file_path=output_path)
    print(f"✅ Audio file saved to: {output_path}")
    
    return output_folder

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_to_speech.py \"Your text to synthesize\"")
        sys.exit(1)
    
    input_text = sys.argv[1]
    folder_path = text_to_speech(input_text)
    print(f"Output folder: {folder_path}")
