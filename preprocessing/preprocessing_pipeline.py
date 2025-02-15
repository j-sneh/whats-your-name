import os
import cv2
from preprocessing.audio_processing import process_video
from preprocessing.face_detection import extract_face_embedding
import database
import json
import postprocessing.text_to_speech as tts

def preprocess_video(video_path, database):
    """
    Preprocesses the video file to extract audio context and face embeddings.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        dict: A dictionary containing the extracted information.
            {
                "name": str,  # Name of the person
                "context": str,  # Context of the conversation
                "face_embedding": np.array,  # Face embedding (key)
                "video_path": str  # Path to the video file
            }
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    # Step 1: Extract audio information (name and context)
    tkt = process_video(video_path=video_path)
    json_data = json.dumps(tts.extract_info_claude(tkt))
    name = json_data["name"]
    context = json_data["other_info"]
    # Step 2: Extract face embedding from the video
    face_embedding = extract_face_embedding(video_path)


    # Step 3: Return the extracted information
    return {
        "name": name,
        "context": context,
        "face_embedding": face_embedding,
        "video_path": video_path
    }

if __name__ == "__main__":
    # Example usage
    video_path = "path/to/video.mp4"
    result = preprocess_video(video_path)
    print("Extracted Information:", result)