"""
Anti-Spoofing Web Interface
Streamlit-based web application for face anti-spoofing detection
Supports both image upload and live webcam detection
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess
import threading
from PIL import Image
import time
import queue
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import streamlit_webrtc as webrtc
from av import VideoFrame

# Page configuration
st.set_page_config(
    page_title="Face Anti-Spoofing Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.result-real {
    background-color: #d4edda;
    border: 2px solid #28a745;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.result-fake {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.result-uncertain {
    background-color: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.confidence-bar {
    height: 25px;
    border-radius: 12px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Global variable for the video processor instance
video_processor_instance = None

@st.cache_resource
def load_models():
    """Load the face detection and anti-spoofing models"""
    try:
        face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
        anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')
        return face_detector, anti_spoof, None
    except Exception as e:
        return None, None, str(e)

def increased_crop(img, bbox, bbox_inc=1.5):
    """Crop face region with increased bounding box"""
    real_h, real_w = img.shape[:2]

    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)

    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)

    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img,
                             y1-y, int(l*bbox_inc-y2+y),
                             x1-x, int(l*bbox_inc)-x2+x,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def predict_image(img, face_detector, anti_spoof):
    """Make prediction on uploaded image"""
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.array(img)

    # Detect faces
    bbox_result = face_detector([img_array])[0]

    if bbox_result.shape[0] == 0:
        return None, None, None, None, "No face detected"

    # Get the first face
    bbox = bbox_result.flatten()[:4].astype(int)

    # Crop face for anti-spoofing
    cropped_face = increased_crop(img_array, bbox, bbox_inc=1.5)

    # Make prediction
    pred = anti_spoof([cropped_face])[0]

    # Extract probabilities
    real_confidence = pred[0][0]
    fake_confidence = pred[0][1] if len(pred[0]) > 1 else 1 - real_confidence

    # Determine label
    label = np.argmax(pred[0])

    return bbox, label, real_confidence, fake_confidence, None

def draw_bounding_box(img, bbox, label, real_conf, fake_conf, threshold=0.5):
    """Draw bounding box and prediction on image"""
    img_array = np.array(img)

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        # Determine color
        if label == 0 and real_conf > threshold:
            color = (0, 255, 0)  # Green for real
            text = f"REAL: {real_conf:.3f}"
        elif label == 0:
            color = (255, 165, 0)  # Orange for uncertain
            text = f"UNCERTAIN: {real_conf:.3f}"
        else:
            color = (255, 0, 0)  # Red for fake
            text = f"FAKE: {fake_conf:.3f}"

        # Draw bounding box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)

        # Add text
        cv2.putText(img_array, text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return Image.fromarray(img_array)

def create_confidence_bars(real_conf, fake_conf):
    """Create HTML confidence bars"""
    real_percentage = real_conf * 100
    fake_percentage = fake_conf * 100

    html = f"""
    <div style="margin: 1rem 0;">
        <div style="margin-bottom: 0.5rem;">
            <strong>Real Face Confidence: {real_percentage:.1f}%</strong>
            <div style="background-color: #e0e0e0; border-radius: 12px; height: 25px;">
                <div class="confidence-bar" style="width: {real_percentage}%; background-color: #28a745;"></div>
            </div>
        </div>
        <div>
            <strong>Fake Face Confidence: {fake_percentage:.1f}%</strong>
            <div style="background-color: #e0e0e0; border-radius: 12px; height: 25px;">
                <div class="confidence-bar" style="width: {fake_percentage}%; background-color: #dc3545;"></div>
            </div>
        </div>
    </div>
    """
    return html

class VideoProcessor:
    def __init__(self):
        self.face_detector, self.anti_spoof, self.error = load_models()
        self.threshold = 0.5
        self.stats = {
            'total_frames': 0,
            'real_count': 0,
            'fake_count': 0,
            'uncertain_count': 0,
            'no_face_count': 0
        }

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox_result = self.face_detector([rgb_img])[0]
        self.stats['total_frames'] += 1
        if bbox_result.shape[0] == 0:
            self.stats['no_face_count'] += 1
            return frame
        bbox = bbox_result.flatten()[:4].astype(int)
        cropped_face = increased_crop(rgb_img, bbox, bbox_inc=1.5)
        pred = self.anti_spoof([cropped_face])[0]
        real_confidence = pred[0][0]
        fake_confidence = pred[0][1] if len(pred[0]) > 1 else 1 - real_confidence
        label = np.argmax(pred[0])
        if label == 0 and real_confidence > self.threshold:
            self.stats['real_count'] += 1
            color = (0, 255, 0)
            text = f"REAL: {real_confidence:.3f}"
        elif label == 0:
            self.stats['uncertain_count'] += 1
            color = (255, 165, 0)
            text = f"UNCERTAIN: {real_confidence:.3f}"
        else:
            self.stats['fake_count'] += 1
            color = (0, 0, 255)
            text = f"FAKE: {fake_confidence:.3f}"
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Face Anti-Spoofing Detection</h1>', unsafe_allow_html=True)

    # Main content (only live webcam detection)
    st.header("Live Webcam Detection (Real-Time)")

    # Load models
    with st.spinner("Loading AI models..."):
        face_detector, anti_spoof, error = load_models()

    if error:
        st.error(f"❌ Error loading models: {error}")
        st.stop()

    st.success("✅ Models loaded successfully!")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values = more strict detection"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.markdown("""
    This application uses advanced AI to detect face spoofing attacks:
    - **Real Face**: Live person detected
    - **Fake Face**: Photo, video, or mask detected
    - **Uncertain**: Low confidence prediction
    """)

    # Use global variable for video processor
    global video_processor_instance
    if video_processor_instance is None:
        video_processor_instance = VideoProcessor()
    video_processor_instance.threshold = threshold

    webrtc_streamer = webrtc.webrtc_streamer(
        key="anti-spoof-live",
        mode=webrtc.WebRtcMode.SENDRECV,
        video_processor_factory=lambda: video_processor_instance,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Show stats
    stats = video_processor_instance.stats
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    with stats_col1:
        st.metric("Total Frames", stats['total_frames'])
    with stats_col2:
        st.metric("Real Faces", stats['real_count'], delta=None)
    with stats_col3:
        st.metric("Fake Faces", stats['fake_count'], delta=None)
    with stats_col4:
        st.metric("Uncertain", stats['uncertain_count'], delta=None)
    with stats_col5:
        st.metric("No Face", stats['no_face_count'], delta=None)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("### 🛡️ Security Notice")
        st.info("This tool is for educational and testing purposes. For production security systems, additional validation and testing is recommended.")

if __name__ == "__main__":
    main()
