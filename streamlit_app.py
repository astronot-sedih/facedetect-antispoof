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
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

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

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Face Anti-Spoofing Detection</h1>', unsafe_allow_html=True)
    
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
    
    # Main content
    tab1, tab2 = st.tabs(["📸 Image Upload Detection", "🎥 Live Webcam Detection"])
    
    # Tab 1: Image Upload
    with tab1:
        st.header("Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a photo containing a face to test"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                bbox, label, real_conf, fake_conf, error = predict_image(
                    image, face_detector, anti_spoof
                )
            
            with col2:
                st.subheader("Detection Result")
                
                if error:
                    st.warning(f"⚠️ {error}")
                else:
                    # Draw result image
                    result_image = draw_bounding_box(
                        image, bbox, label, real_conf, fake_conf, threshold
                    )
                    st.image(result_image, use_column_width=True)
                    
                    # Display results
                    if label == 0 and real_conf > threshold:
                        st.markdown(
                            f'<div class="result-real"><h3>✅ REAL FACE DETECTED</h3><p>This appears to be a live person.</p></div>',
                            unsafe_allow_html=True
                        )
                    elif label == 0:
                        st.markdown(
                            f'<div class="result-uncertain"><h3>⚠️ UNCERTAIN</h3><p>Low confidence - could be real or fake.</p></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="result-fake"><h3>❌ SPOOFING DETECTED</h3><p>This appears to be a photo, video, or mask.</p></div>',
                            unsafe_allow_html=True
                        )
                    
                    # Confidence bars
                    st.markdown("### Confidence Scores")
                    confidence_html = create_confidence_bars(real_conf, fake_conf)
                    st.markdown(confidence_html, unsafe_allow_html=True)
                      # Detailed metrics
                    with st.expander("📊 Detailed Metrics"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Real Confidence", f"{real_conf:.4f}")
                            st.metric("Prediction Label", "Real" if label == 0 else "Fake")
                        with col_b:
                            st.metric("Fake Confidence", f"{fake_conf:.4f}")
                            if bbox is not None:
                                st.metric("Face Coordinates", f"({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                            else:
                                st.metric("Face Coordinates", "Not detected")
      # Tab 2: Live Webcam
    with tab2:
        st.header("Live Webcam Detection")
        
        # Status check
        st.markdown("### 📊 System Status")
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            # Check if webcam is available
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    st.success("✅ Webcam Available")
                    cap.release()
                else:
                    st.error("❌ Webcam Not Available")
            except:
                st.error("❌ Webcam Error")
        
        with col_status2:
            # Check if live script exists
            if os.path.exists("live_webcam_test.py"):
                st.success("✅ Live Detection Script Ready")
            else:
                st.error("❌ Live Detection Script Missing")
        
        st.markdown("---")
        st.info("📌 For live webcam detection, use the command line script for better performance:")
        
        st.code("python live_webcam_test.py", language="bash")
        
        st.markdown("### 🎮 Available Commands:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Basic Usage:**
            ```bash
            python live_webcam_test.py
            ```
            
            **Custom Threshold:**
            ```bash
            python live_webcam_test.py --threshold 0.7
            ```
            """)
        
        with col2:
            st.markdown("""
            **Save Video:**
            ```bash
            python live_webcam_test.py --save_video output.avi
            ```
            
            **Different Camera:**
            ```bash
            python live_webcam_test.py --camera 1
            ```
            """)
        
        st.markdown("### 📱 Live Detection Features:")
        st.markdown("""
        - **Real-time processing** with your webcam
        - **Interactive controls** (Q to quit, S to save frame, R to reset stats)
        - **Live statistics** showing detection counts and FPS
        - **Visual feedback** with colored bounding boxes
        - **Video recording** capability
        """)
          # Webcam simulation placeholder        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Launch Live Detection"):
                try:
                    st.info("🎥 Launching live webcam detection...")
                    st.warning("⚠️ A new console window will open for webcam detection.")
                    
                    # Launch using the helper script
                    result = subprocess.Popen([
                        "python", "launch_live_detection.py", 
                        "--threshold", str(threshold)
                    ])
                    
                    st.success("✅ Live detection launched!")
                    st.info("💡 Look for a new console window with webcam feed.")
                    st.info("🎮 Controls: Press 'Q' to quit, 'S' to save frame, 'R' to reset stats")
                    
                except Exception as e:
                    st.error(f"❌ Error launching live detection: {e}")
                    st.code(f"Manual command: python live_webcam_test.py --threshold {threshold}")
        
        with col2:
            if st.button("🎬 Launch with Video Recording"):
                try:
                    timestamp = int(time.time())
                    video_filename = f"live_detection_{timestamp}.avi"
                    
                    st.info(f"🎥 Launching live detection with recording...")
                    st.warning("⚠️ A new console window will open.")
                    
                    # Launch with video recording
                    subprocess.Popen([
                        "python", "launch_live_detection.py", 
                        "--threshold", str(threshold),
                        "--save_video", video_filename
                    ])
                    
                    st.success("✅ Live detection with recording launched!")
                    st.info(f"� Video will be saved as: {video_filename}")
                    
                except Exception as e:
                    st.error(f"❌ Error launching live detection: {e}")
                    st.code(f"Manual command: python live_webcam_test.py --threshold {threshold} --save_video output.avi")
        
        # Manual command section
        st.markdown("### 🖥️ Manual Command Line Options:")
        
        command_basic = f"python live_webcam_test.py --threshold {threshold}"
        command_record = f"python live_webcam_test.py --threshold {threshold} --save_video output.avi"
        command_camera = f"python live_webcam_test.py --threshold {threshold} --camera 1"
        
        st.code(f"# Basic live detection\n{command_basic}", language="bash")
        st.code(f"# With video recording\n{command_record}", language="bash")
        st.code(f"# Different camera\n{command_camera}", language="bash")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("### 🛡️ Security Notice")
        st.info("This tool is for educational and testing purposes. For production security systems, additional validation and testing is recommended.")

if __name__ == "__main__":
    main()
