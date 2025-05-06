import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CosineSimilarity
import os
import re 
MODEL_PATH = r'Family-embeded.keras'
model = load_model(MODEL_PATH)
# if "model" not in st.session_state:
#     st.session_state["model"] = load_model(MODEL_PATH)
# model =  st.session_state["model"]

# Configure MediaPipe
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

# Threshold
THRESHOLD = 0.5

# Define functions
def get_frame_landmarks(frame):
    """Extract landmarks from a frame."""
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

def extract_embedding(video_landmarks):
    """Create an embedding from landmarks."""
    video_landmarks = np.array(video_landmarks)
    target_shape = (120, 100, 3)

    if video_landmarks.shape[0] < target_shape[0]:
        padding = target_shape[0] - video_landmarks.shape[0]
        video_landmarks = np.pad(video_landmarks, ((0, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        video_landmarks = video_landmarks[:target_shape[0], :, :]

    video_landmarks = np.reshape(video_landmarks, (1, *target_shape))
    embedding = model.predict(video_landmarks)
    return embedding

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate Cosine Similarity between two embeddings."""
    cosine_similarity = CosineSimilarity()
    return cosine_similarity(embedding1, embedding2).numpy()



video_list = []
with open("output_filenames.txt", "r", encoding="utf-8") as f:
    for line in f:
        filename = line.strip()
        if filename:
            video_list.append(os.path.join("Family_video", filename))
embedding_dir = "reference_embedding2"
# for video_path in video_list:
  
#     video_name = os.path.basename(video_path).split('.')[0]
#     embedding_path = os.path.join(embedding_dir, f"{video_name}_embedding.npy")

   
#     if not os.path.exists(embedding_path):
#         cap = cv2.VideoCapture(video_path)
#         ref_landmarks = []

      
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Gọi hàm lấy landmarks
#             landmarks = get_frame_landmarks(frame_rgb)
#             if landmarks is not None:
#                 ref_landmarks.append(landmarks)

#         cap.release()

       
#         if ref_landmarks: 
#             reference_embedding = extract_embedding(ref_landmarks)
#             np.save(embedding_path, reference_embedding)  
#             print(f"Embedding saved: {embedding_path}")
#         else:
#             print(f"No landmarks detected for video: {video_name}")
#     else:
#         print(f"Embedding already exists: {embedding_path}")


st.set_page_config(page_title="Sign Language Learning", layout="wide")

if "match_detected" not in st.session_state:
    st.session_state["match_detected"] = False

# Hàm xử lý tên bài học và nhãn
def clean_label(video_path):
    video_name = os.path.basename(video_path).split('.')[0]
    raw_label_name = video_name.split('-')[-1]  # Lấy phần sau dấu '-'
    return re.sub(r'\s*\d+$', '', raw_label_name)  # Loại bỏ số ở cuối nhãn nếu có

# Sidebar with Roadmap
st.sidebar.title("Learning Roadmap")
roadmap_videos = {
    "Chapter 1: Chào hỏi và giao tiếp cơ bản": video_list[:15],
    "Chapter 2: Gia đình và con người": video_list[15:31],
    "Chapter 3: Hành động và hoạt động hàng ngày": video_list[31:47],
    "Chapter 4: Số đếm, màu sắc, cảm xúc": video_list[47:62],
    "Chapter 5: Đồ vật, địa điểm, thực phẩm": video_list[62:]
}

# Danh sách bài học đã xử lý
selected_chapter = st.sidebar.selectbox("Select a Chapter", list(roadmap_videos.keys()))
chapter_videos = roadmap_videos[selected_chapter]
lesson_names = [clean_label(video) for video in chapter_videos]  # Tên bài học không chứa số cuối

# Hiển thị bài học
selected_lesson_name = st.sidebar.selectbox("Select a Lesson", lesson_names)
selected_lesson_index = lesson_names.index(selected_lesson_name)
lesson_video_path = chapter_videos[selected_lesson_index]

# Lấy nhãn bài học hiện tại
label_name = clean_label(lesson_video_path)
embedding_path = os.path.join(embedding_dir, f"{os.path.basename(lesson_video_path).split('.')[0]}_embedding.npy")


# Reset WebRTC streamer khi chọn bài học mới
if "webrtc_key" not in st.session_state:
    st.session_state.webrtc_key = "initial_key"

# Tạo khóa mới khi bài học thay đổi
if "previous_lesson" not in st.session_state or st.session_state.previous_lesson != (selected_chapter, selected_lesson_name):
    st.session_state.previous_lesson = (selected_chapter, selected_lesson_name)
    st.session_state.match_detected = False  # Reset trạng thái nhận diện
    st.session_state.webrtc_key = f"webrtc_{selected_chapter}_{selected_lesson_name}"  # Tạo khóa mới để reset camera

reference_embedding = np.load(embedding_path)

col1, col2 = st.columns(2)

with col1:
    st.title("Example")
    st.video(lesson_video_path)
    st.subheader(f"Lesson: {label_name}")  # Hiển thị tên nhãn hiện tại

with col2:
    st.title("Practice")

    class SignLanguageTransformer(VideoTransformerBase):
        def __init__(self):
            self.reference_embedding = reference_embedding
            self.user_landmarks = []
            self.match_detected = False

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks = get_frame_landmarks(img_rgb)
            self.user_landmarks.append(landmarks)

            if len(self.user_landmarks) > 60 and not self.match_detected:  # Process 60 frames
                user_embedding = extract_embedding(self.user_landmarks)
                similarity = calculate_cosine_similarity(self.reference_embedding, user_embedding)

                if similarity > THRESHOLD:
                    self.match_detected = True
                    st.session_state.match_detected = True  # Cập nhật trạng thái

                self.user_landmarks = []  

            if self.match_detected:
                cv2.putText(img, "Match!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Keep Practicing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return img


    # Khởi động WebRTC streamer với khóa mới
    webrtc_streamer(key=st.session_state.webrtc_key, video_processor_factory=SignLanguageTransformer)

if st.session_state.match_detected:
    st.success("Chúc mừng! Bạn đã thực hiện đúng cử chỉ!")
