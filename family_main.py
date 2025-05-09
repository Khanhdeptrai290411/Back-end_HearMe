import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import base64
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CosineSimilarity

# ============ CẤU HÌNH ============
MODEL_PATH = "Family/Family-embeded.keras"
VIDEO_DIR = "Family/Family_video2"
EMBEDDING_DIR = "Family/reference_embedding2"
FILENAME_TXT = "Family/output_filenames.txt"
FRAMES_LIMIT = 60
SIMILARITY_THRESHOLD = 0.5

# ============ KHỞI TẠO ============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="public"), name="public")
model = load_model(MODEL_PATH)

# ============ MEDIA PIPE ============
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(False, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [4, 6, 8, 9, 33, 37, 40, 46, 52, 55, 61, 70, 80, 82, 84, 87, 88, 91,
                 105, 107, 133, 145, 154, 157, 159, 161, 163, 263, 267, 270, 276,
                 282, 285, 291, 300, 310, 312, 314, 317, 318, 321, 334, 336, 362,
                 374, 381, 384, 386, 388, 390, 468, 473]
HAND_NUM, POSE_NUM, FACE_NUM = len(filtered_hand), len(filtered_pose), len(filtered_face)

# ============ XÂY DỰNG ROADMAP ============
def clean_label(video_path):
    video_name = os.path.basename(video_path).split('.')[0]
    raw_label_name = video_name.split('-')[-1]
    return re.sub(r'\s*\d+$', '', raw_label_name)

video_list = []
with open(FILENAME_TXT, "r", encoding="utf-8") as f:
    for line in f:
        filename = line.strip()
        if filename:
            full_path = os.path.join(VIDEO_DIR, filename)
            video_list.append(full_path)

# Tự chia chương mỗi 15 video
roadmap = {}
for i in range(0, len(video_list), 15):
    chapter_name = f"Chapter {i // 15 + 1}"
    chapter = []
    for vid in video_list[i:i + 15]:
        base = os.path.basename(vid).split('.')[0]
        label = clean_label(vid)
        public_path = f"/Family_video2/{os.path.basename(vid)}"
        embedding_path = f"{EMBEDDING_DIR}/{base}_embedding.npy".replace("\\", "/")
        chapter.append({"name": label, "path": public_path, "embedding": embedding_path})
    roadmap[chapter_name] = chapter

# ============ LANDMARKS ============
def get_frame_landmarks(frame):
    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))

    results_hands = hands.process(frame)
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            index = 0 if results_hands.multi_handedness[i].classification[0].index == 0 else HAND_NUM
            all_landmarks[index:index+HAND_NUM, :] = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    results_pose = pose.process(frame)
    if results_pose.pose_landmarks:
        all_landmarks[HAND_NUM * 2:HAND_NUM * 2 + POSE_NUM, :] = np.array(
            [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark])[filtered_pose]

    results_face = face_mesh.process(frame)
    if results_face.multi_face_landmarks:
        all_landmarks[HAND_NUM * 2 + POSE_NUM:, :] = np.array(
            [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark])[filtered_face]

    return all_landmarks

def extract_embedding(video_landmarks):
    video_landmarks = np.array(video_landmarks)
    target_shape = (120, 100, 3)
    if video_landmarks.shape[0] < target_shape[0]:
        padding = target_shape[0] - video_landmarks.shape[0]
        video_landmarks = np.pad(video_landmarks, ((0, padding), (0, 0), (0, 0)), mode='constant')
    else:
        video_landmarks = video_landmarks[:target_shape[0], :, :]
    video_landmarks = np.reshape(video_landmarks, (1, *target_shape))
    return model.predict(video_landmarks)

def calculate_cosine_similarity(embedding1, embedding2):
    cosine_similarity = CosineSimilarity()
    return cosine_similarity(embedding1, embedding2).numpy()

# ============ MODEL ============
class VideoProcessRequest(BaseModel):
    frames: List[str]
    lessonPath: str

@app.get("/api/roadmap")
async def get_roadmap():
    return roadmap

@app.post("/api/process-video")
async def process_video(request: VideoProcessRequest):
    frames = request.frames
    lesson_path = request.lessonPath

    reference_embedding_path = None
    for chapter in roadmap.values():
        for lesson in chapter:
            if lesson["path"] == lesson_path:
                reference_embedding_path = lesson["embedding"]
                break
        if reference_embedding_path:
            break

    if not reference_embedding_path or not os.path.exists(reference_embedding_path):
        raise HTTPException(status_code=400, detail="Lesson not found or embedding missing")

    user_landmarks = []
    for frame_data in frames[:FRAMES_LIMIT]:
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = get_frame_landmarks(frame_rgb)
        user_landmarks.append(landmarks)

    user_embedding = extract_embedding(user_landmarks)
    reference_embedding = np.load(reference_embedding_path)

    similarity = calculate_cosine_similarity(user_embedding, reference_embedding)

    return {
        "similarity": float(similarity),
        "status": "Match!" if similarity > SIMILARITY_THRESHOLD else "Keep Practicing"
    }

@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/public/index.html")
