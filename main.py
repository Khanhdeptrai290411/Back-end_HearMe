from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CosineSimilarity
import base64
import json

app = FastAPI()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ file tĩnh (HTML, video)
app.mount("/public", StaticFiles(directory="public"), name="public")

# Cấu hình MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

# Landmark filters
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [4, 6, 8, 9, 33, 37, 40, 46, 52, 55, 61, 70, 80, 82, 84, 87, 88, 91, 105, 107, 133, 145, 154, 157, 159, 161, 163, 263, 267, 270, 276, 282, 285, 291, 300, 310, 312, 314, 317, 318, 321, 334, 336, 362, 374, 381, 384, 386, 388, 390, 468, 473]
HAND_NUM, POSE_NUM, FACE_NUM = len(filtered_hand), len(filtered_pose), len(filtered_face)

# Đường dẫn mô hình
MODEL_PATH = "Color_model/color-embeded-acc.h5"
model = load_model(MODEL_PATH)

# Danh sách chương và bài học
roadmap = {
    "Chapter 1": [
        {"name": "FLOWER", "path": "/public/Color_video/509912909642189-FLOWER.mp4", "embedding": "reference_embedding/509912909642189-FLOWER_embedding.npy"},
        {"name": "CONGRATULATIONS", "path": "/public/Color_video/164951232112037-CONGRATULATIONS.mp4", "embedding": "reference_embedding/164951232112037-CONGRATULATIONS_embedding.npy"},
        {"name": "WELCOME", "path": "/public/Color_video/12579272885288595-WELCOME.mp4", "embedding": "reference_embedding/12579272885288595-WELCOME_embedding.npy"},
        {"name": "HELLO", "path": "/public/Color_video/5339916560192981-HELLO.mp4", "embedding": "reference_embedding/5339916560192981-HELLO_embedding.npy"},
        {"name": "DESIGN", "path": "/public/Color_video/7119068123432775-DESIGN.mp4", "embedding": "reference_embedding/7119068123432775-DESIGN_embedding.npy"}
    ],
    "Chapter 2": [
        {"name": "BRIGHT", "path": "/public/Color_video/18880274572777278-BRIGHT.mp4", "embedding": "reference_embedding/18880274572777278-BRIGHT_embedding.npy"},
        {"name": "GREET", "path": "/public/Color_video/25196207384020064-GREET.mp4", "embedding": "reference_embedding/25196207384020064-GREET_embedding.npy"},
        {"name": "CHEER", "path": "/public/Color_video/48093054817466707-CHEER.mp4", "embedding": "reference_embedding/48093054817466707-CHEER_embedding.npy"}
    ]
}

# Hàm lấy landmarks từ frame
def get_frame_landmarks(frame):
    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))

    def get_hands(frame):
        results_hands = hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].index == 0:  # Right hand
                    all_landmarks[:HAND_NUM, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                else:
                    all_landmarks[HAND_NUM:HAND_NUM * 2, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    def get_pose(frame):
        results_pose = pose.process(frame)
        if results_pose.pose_landmarks:
            all_landmarks[HAND_NUM * 2:HAND_NUM * 2 + POSE_NUM, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark])[filtered_pose]

    def get_face(frame):
        results_face = face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            all_landmarks[HAND_NUM * 2 + POSE_NUM:, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark])[filtered_face]

    get_hands(frame)
    get_pose(frame)
    get_face(frame)

    return all_landmarks

# Hàm tạo embedding
def extract_embedding(video_landmarks):
    video_landmarks = np.array(video_landmarks)
    target_shape = (60, 100, 3)

    if video_landmarks.shape[0] < target_shape[0]:
        padding = target_shape[0] - video_landmarks.shape[0]
        video_landmarks = np.pad(video_landmarks, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        video_landmarks = video_landmarks[:target_shape[0], :, :]

    video_landmarks = np.reshape(video_landmarks, (1, *target_shape))
    embedding = model.predict(video_landmarks)
    return embedding

# Hàm tính cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    cosine_similarity = CosineSimilarity()
    return cosine_similarity(embedding1, embedding2).numpy()

# Model cho request body
class VideoProcessRequest(BaseModel):
    frames: List[str]  # Danh sách frame ở định dạng base64
    lessonPath: str    # Đường dẫn bài học

# API để lấy roadmap
@app.get("/api/roadmap")
async def get_roadmap():
    return roadmap

# API để xử lý video
@app.post("/api/process-video")
async def process_video(request: VideoProcessRequest):
    frames = request.frames
    lesson_path = request.lessonPath

    # Tìm embedding tham chiếu
    reference_embedding_path = None
    for chapter in roadmap.values():
        for lesson in chapter:
            if lesson["path"] == lesson_path:
                reference_embedding_path = lesson["embedding"]
                break
        if reference_embedding_path:
            break

    if not reference_embedding_path:
        raise HTTPException(status_code=400, detail="Lesson not found")

    # Xử lý frame
    user_landmarks = []
    for frame_data in frames[:60]:  # Giới hạn 60 frame
        # Decode base64 thành frame
        img_data = base64.b64decode(frame_data.split(',')[1])  # Bỏ phần "data:image/jpeg;base64,"
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Lấy landmarks
        landmarks = get_frame_landmarks(frame_rgb)
        user_landmarks.append(landmarks)

    # Tạo embedding cho video người dùng
    user_embedding = extract_embedding(user_landmarks)

    # Tải embedding tham chiếu
    reference_embedding = np.load(reference_embedding_path)

    # Tính độ tương đồng
    similarity = calculate_cosine_similarity(user_embedding, reference_embedding)

    # Trả về kết quả
    return {
        "similarity": float(similarity),
        "status": "Match!" if similarity > 0.8 else "Keep Practicing"
    }

# Redirect root URL đến index.html
@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/public/index.html")