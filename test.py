import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import mysql.connector
import numpy as np
import cv2
import base64
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CosineSimilarity

# ============ CẤU HÌNH ============
FRAMES_LIMIT = 60  # Giữ cố định vì không có trong cơ sở dữ liệu

# ============ KHỞI TẠO FASTAPI ============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="public"), name="public")

# ============ KẾT NỐI MYSQL ============
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="hearme_learning"
        )
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# ============ TẢI CẤU HÌNH TỪ DATABASE ============
# ============ TẢI CẤU HÌNH TỪ DATABASE ============
def load_config_from_db(model_id=1):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT model_file, embedding_dir, threshold, target_shape
            FROM models
            WHERE id = %s
        """, (model_id,))
        config = cursor.fetchone()
        if not config:
            raise HTTPException(status_code=500, detail="Model not found in database")
        # Chuyển target_shape từ chuỗi '(120, 100, 3)' thành tuple (120, 100, 3)
        target_shape_str = config['target_shape'].strip("()").replace(" ", "")
        try:
            target_shape = tuple(int(x) for x in target_shape_str.split(","))
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Invalid target_shape format: {str(e)}")
        return {
            "model_file": config['model_file'],
            "embedding_dir": config['embedding_dir'],
            "threshold": config['threshold'],
            "target_shape": target_shape,
            "video_dir": "Family/Family_video2"  # Giả định tạm thời
        }
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Load cấu hình và mô hình
config = load_config_from_db()
model = load_model(config['model_file'])

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

# ============ HÀM XỬ LÝ ROADMAP TỪ DATABASE ============
def clean_label(video_filename):
    raw_label_name = video_filename.split('-')[-1].split('.')[0]
    return re.sub(r'\s*\d+$', '', raw_label_name)

def get_roadmap_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Lấy tất cả chapters thuộc model_id = 1
        cursor.execute("""
            SELECT id, name
            FROM chapters
            WHERE model_id = 1
            ORDER BY id
        """)
        chapters = cursor.fetchall()

        roadmap = {}
        for chapter in chapters:
            chapter_id = chapter['id']
            chapter_name = chapter['name']

            # Lấy tất cả videos thuộc chapter
            cursor.execute("""
                SELECT video_filename
                FROM videos
                WHERE model_id = 1 AND chapter_id = %s
            """, (chapter_id,))
            videos = cursor.fetchall()

            chapter_videos = []
            for video in videos:
                video_filename = video['video_filename']
                base = video_filename.split('.')[0]
                label = clean_label(video_filename)
                public_path = f"/{config['video_dir']}/{video_filename}".replace("Family/", "")
                embedding_path = f"{config['embedding_dir']}/{base}_embedding.npy".replace("\\", "/")
                chapter_videos.append({
                    "name": label,
                    "path": public_path,
                    "embedding": embedding_path
                })

            roadmap[chapter_name] = chapter_videos

        return roadmap
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

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
    return get_roadmap_from_db()

@app.post("/api/process-video")
async def process_video(request: VideoProcessRequest):
    frames = request.frames
    lesson_path = request.lessonPath

    # Tìm embedding path từ roadmap trong database
    roadmap = get_roadmap_from_db()
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

    # Xử lý frames từ client
    user_landmarks = []
    for frame_data in frames[:FRAMES_LIMIT]:
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = get_frame_landmarks(frame_rgb)
        user_landmarks.append(landmarks)

    # Tạo embedding từ video của người dùng
    user_embedding = extract_embedding(user_landmarks)
    reference_embedding = np.load(reference_embedding_path)

    # Tính độ tương đồng
    similarity = calculate_cosine_similarity(user_embedding, reference_embedding)

    return {
        "similarity": float(similarity),
        "status": "Match!" if similarity > config['threshold'] else "Keep Practicing"
    }

@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/public/index.html")