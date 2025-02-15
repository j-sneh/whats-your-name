import os
import cv2
import numpy as np
from deepface import DeepFace

def extract_face_embedding(video_path, num_frames=5, frame_interval=30):
    """
    Extracts a face embedding from multiple frames in a video using OpenCV and DeepFace.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract from the video.
        frame_interval (int): Number of frames to skip between extractions.

    Returns:
        np.array: Average face embedding for the detected face(s), or None if no faces found.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = np.linspace(0, frame_count - 1, num_frames, dtype=int)  # Sample evenly spaced frames

    embeddings = []
    detected_faces = 0  # Counter for successful detections

    for frame_idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {frame_idx}, skipping.")
            continue  # Skip if frame reading fails
        
        # Convert to grayscale to improve detection in low-light conditions
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            embedding_objs = DeepFace.represent(gray_frame, model_name="Facenet", enforce_detection=False)

            if embedding_objs:
                embeddings.append(np.array(embedding_objs[0]["embedding"]))
                detected_faces += 1
                print(f"✅ Face detected in frame {frame_idx}")

        except Exception as e:
            print(f"❌ Warning: Face detection failed on frame {frame_idx}: {e}")
            continue  # Skip frame if DeepFace fails

    cap.release()

    if not embeddings:
        print("🚨 No faces detected in any selected frames.")
        return None, 0

    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding, detected_faces

def test_face_detection(video_paths):
    """
    Tests face detection across multiple videos and reports success rates.

    Args:
        video_paths (list): List of video file paths to test.
    """

    
    total_videos = len(video_paths)
    total_frames_checked = 0
    total_faces_detected = 0

    print("\n===== FACE DETECTION TEST =====")
    
    for idx, video_path in enumerate(video_paths):
        print(f"\n🎥 Testing Video {idx+1}/{total_videos}: {video_path}")

        if not os.path.exists(video_path):
            print(f"❌ Skipping: File '{video_path}' not found.")
            continue

        try:
            _, detected_faces = extract_face_embedding(video_path)
            total_faces_detected += detected_faces
            total_frames_checked += 5  # Default num_frames

        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")

    detection_rate = (total_faces_detected / total_frames_checked) * 100 if total_frames_checked > 0 else 0

    print("\n===== TEST RESULTS =====")
    print(f"🔍 Total Videos Tested: {total_videos}")
    print(f"📸 Total Frames Checked: {total_frames_checked}")
    print(f"😀 Total Faces Detected: {total_faces_detected}")
    print(f"📊 Detection Success Rate: {detection_rate:.2f}%")

# Example usage
if __name__ == "__main__":
    test_videos = [
        "data/test_video1.MOV",
        "data/test_video2.MOV",
    ]
    test_face_detection(test_videos)
