import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit as st  # Chỉ giữ nếu dùng Streamlit

# Configure MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# Landmark filters
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [4, 6, 8, 9, 33, 37, 40, 46, 52, 55, 61, 70, 80, 82, 84, 87, 88, 91, 105, 107, 133, 145, 154, 157, 159, 161, 163, 263, 267, 270, 276, 282, 285, 291, 300, 310, 312, 314, 317, 318, 321, 334, 336, 362, 374, 381, 384, 386, 388, 390, 468, 473]
HAND_NUM, POSE_NUM, FACE_NUM = len(filtered_hand), len(filtered_pose), len(filtered_face)

# Load model
MODEL_PATH = r'Family-embeded.keras'
if "model" not in st.session_state:
    st.session_state["model"] = load_model(MODEL_PATH)
model = st.session_state["model"]

def initialize_mediapipe():
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)
    return hands, pose, face_mesh

def get_frame_landmarks(frame, hands, pose, face_mesh):
    if frame is None or frame.size == 0:
        return None
    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hands
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            if results_hands.multi_handedness[i].classification[0].index == 0:  # Right hand
                all_landmarks[:HAND_NUM, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            else:
                all_landmarks[HAND_NUM:HAND_NUM * 2, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    # Pose
    results_pose = pose.process(frame_rgb)
    if results_pose.pose_landmarks:
        all_landmarks[HAND_NUM * 2:HAND_NUM * 2 + POSE_NUM, :] = np.array(
            [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark])[filtered_pose]

    # Face
    results_face = face_mesh.process(frame_rgb)
    if results_face.multi_face_landmarks:
        all_landmarks[HAND_NUM * 2 + POSE_NUM:, :] = np.array(
            [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark])[filtered_face]

    return all_landmarks

def extract_embedding(video_landmarks, target_shape=(120, 100, 3)):
    video_landmarks = np.array(video_landmarks)
    if video_landmarks.shape[0] < target_shape[0]:
        padding = target_shape[0] - video_landmarks.shape[0]
        video_landmarks = np.pad(video_landmarks, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        video_landmarks = video_landmarks[:target_shape[0], :, :]
    video_landmarks = np.reshape(video_landmarks, (1, *target_shape))
    embedding = model.predict(video_landmarks)
    return embedding

# 💡 Load video list from output_filenames.txt
video_list = []
with open("output_filenames.txt", "r", encoding="utf-8") as f:
    for line in f:
        filename = line.strip()
        if filename:
            video_list.append(os.path.join("Family_video", filename))

# Embedding extraction
embedding_dir = "reference_embedding2"
os.makedirs(embedding_dir, exist_ok=True)

hands, pose, face_mesh = initialize_mediapipe()
try:
    for video_path in video_list:
        video_name = os.path.basename(video_path).split('.')[0]
        embedding_path = os.path.join(embedding_dir, f"{video_name}_embedding.npy")
        if not os.path.exists(embedding_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                continue
            ref_landmarks = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks = get_frame_landmarks(frame, hands, pose, face_mesh)
                if landmarks is not None:
                    ref_landmarks.append(landmarks)
            cap.release()
            if ref_landmarks:
                reference_embedding = extract_embedding(ref_landmarks)
                np.save(embedding_path, reference_embedding)
                print(f"Embedding saved: {embedding_path}")
            else:
                print(f"No landmarks detected for video: {video_name}")
        else:
            print(f"Embedding already exists: {embedding_path}")
finally:
    hands.close()
    pose.close()
    face_mesh.close()
