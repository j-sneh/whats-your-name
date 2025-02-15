import os
import cv2
import numpy as np
from deepface import DeepFace

def cosine_similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two face embeddings.
    
    Args:
        embedding1 (np.array): First face embedding.
        embedding2 (np.array): Second face embedding.

    Returns:
        float: Cosine similarity between the two embeddings.
    """

    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def extract_face_embedding(input_path, num_frames=5, frame_interval=30):
    """
    Extracts a face embedding from an image or multiple frames in a video using OpenCV and DeepFace.

    Args:
        input_path (str): Path to the image or video file.
        num_frames (int): Number of frames to extract from a video.
        frame_interval (int): Number of frames to skip between extractions.

    Returns:
        np.array: Average face embedding for the detected face(s), or None if no faces are found.
    """

    # Check if the input is a video or an image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Error: File '{input_path}' not found.")

    file_ext = os.path.splitext(input_path)[-1].lower()
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

    if file_ext in video_extensions:
        # Process as a video
        return process_video_embeddings(input_path, num_frames, frame_interval)
    else:
        # Process as an image
        return process_image_embedding(input_path)

def process_image_embedding(image_path):
    """
    Extracts a face embedding from a single image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: Face embedding, or None if no face is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Error: Could not read image file {image_path}")

    # Convert to RGB (OpenCV loads images in BGR format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        embedding_objs = DeepFace.represent(rgb_image, model_name="Facenet", enforce_detection=False)
        if embedding_objs:
            print(f"✅ Face detected in image: {image_path}")
            return np.array(embedding_objs[0]["embedding"])
    except Exception as e:
        print(f"❌ Warning: Face detection failed in image: {e}")

    return None

def process_video_embeddings(video_path, num_frames=5, frame_interval=30):
    """
    Extracts face embeddings from multiple frames in a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract from the video.
        frame_interval (int): Number of frames to skip between extractions.

    Returns:
        np.array: Average face embedding for the detected face(s), or None if no faces found.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Error: Unable to open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    embeddings = []
    detected_faces = 0

    for frame_idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Warning: Could not read frame {frame_idx}, skipping.")
            continue  # Skip if frame reading fails

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            embedding_objs = DeepFace.represent(rgb_frame, model_name="Facenet", enforce_detection=False)
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
        return None

    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding


def test_face_detection(video_paths, similarity_threshold=0.9):
    """
    Tests face detection across multiple videos and compares face embeddings.
    
    Args:
        video_paths (list): List of video file paths to test.
        similarity_threshold (float): Cosine similarity threshold for matching faces.
    """

    total_videos = len(video_paths)
    total_frames_checked = 0
    total_faces_detected = 0

    print("\n===== FACE DETECTION TEST =====")

    embeddings = {}  # Dictionary to store embeddings per video
    
    for idx, video_path in enumerate(video_paths):
        print(f"\n🎥 Testing Video {idx+1}/{total_videos}: {video_path}")

        if not os.path.exists(video_path):
            print(f"❌ Skipping: File '{video_path}' not found.")
            continue

        try:
            embedding, detected_faces = extract_face_embedding(video_path)
            if embedding is not None:
                embeddings[video_path] = embedding
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

    # Compare embeddings between videos
    print("\n===== COMPARING FACE EMBEDDINGS =====")

    video_keys = list(embeddings.keys())
    similar_videos = []

    for i in range(len(video_keys)):
        for j in range(i + 1, len(video_keys)):
            video1, video2 = video_keys[i], video_keys[j]

            similarity = cosine_similarity(embeddings[video1], embeddings[video2])

            print(f"🔍 Cosine Similarity between {video1} and {video2}: {similarity:.4f}")

            if similarity > similarity_threshold:
                print(f"✅ Faces in {video1} and {video2} are similar!")
                similar_videos.append((video1, video2))

    if similar_videos:
        print("\n🎉 Similar Faces Found in the Following Videos:")
        for pair in similar_videos:
            print(f"➡ {pair[0]} and {pair[1]}")
    else:
        print("\n❌ No matching faces found across videos.")

# Example usage
if __name__ == "__main__":
    test_videos = [
        "data/test_video1.MOV",
        "data/test_video2.MOV",
        "data/test_video3.MOV"
    ]
    test_face_detection(test_videos)