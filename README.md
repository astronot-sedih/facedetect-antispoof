# Face Anti-Spoofing Detection System

🔒 **Advanced AI-powered face anti-spoofing detection system** that can distinguish between real faces and spoofing attempts (photos, videos, masks, etc.).

## 🚀 Features

- **Real-time webcam detection** with live video feed
- **Video file processing** with frame-by-frame analysis
- **Static image testing** for individual photos
- **Web interface** with Streamlit for easy interaction
- **High accuracy** binary classification (Real vs Fake)
- **Visual feedback** with confidence scores and bounding boxes
- **Multiple model support** with ONNX optimization

## 📋 Requirements

### System Requirements
- Python 3.7+
- Webcam (for live detection)
- Windows OS (batch files optimized for Windows)

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

### 🤖 Model Files
**Download the required model files** and place them in the `saved_models/` directory:

| Model File | Size | Description |
|------------|------|-------------|
| `AntiSpoofing_bin_1.5_128.onnx` | ~50MB | Enhanced anti-spoofing model |
| `AntiSpoofing_bin_128.onnx` | ~30MB | Basic anti-spoofing model |
| `yolov5s-face.onnx` | ~30MB | Face detection model |

**📥 Download Links:**
- **Option 1**: Check the [Releases](https://github.com/Farrellhrs/AntiSpoofingFaceDetection/releases) section
- **Option 2**: Run `bash download_models.sh` (when available)
- **Option 3**: Contact the repository owner for model access

**Required packages:**
- opencv-python
- torch
- torchvision  
- numpy
- onnx
- onnxruntime
- streamlit (for web interface)
- pandas
- tqdm
- tensorboardX
- scikit-learn

## 🗂️ Project Structure

```
AntiSpoofing-FaceDetection/
├── 📁 saved_models/                 # Pre-trained models
│   ├── AntiSpoofing_bin_128.onnx         # Basic binary model
│   ├── AntiSpoofing_bin_1.5_128.onnx     # Enhanced binary model  
│   ├── AntiSpoofing_bin_1.5_128.pth      # PyTorch weights
│   └── yolov5s-face.onnx                 # Face detection model
├── 📁 src/                          # Source code
│   ├── FaceAntiSpoofing.py               # Anti-spoofing model class
│   ├── face_detector/                    # Face detection module
│   │   ├── YOLO.py                      # YOLO face detector
│   │   └── utils.py                     # Utility functions
│   └── [other source files]
├── 📁 sample_test/                  # Sample images and videos
│   ├── Sample_photo_1.jpeg              # Test image 1
│   ├── Sample_photo_2.jpeg              # Test image 2
│   └── Video_Sample_1.mp4               # Test video
├── 📄 streamlit_app.py              # Web interface application
├── 📄 live_webcam_test.py           # Live webcam detection
├── 📄 video_test.py                 # Video processing
├── 📄 test_image.py                 # Image testing
└── 📄 requirements.txt              # Python dependencies
```

## 🎯 Usage Guide

### 1. 🌐 Web Interface (Streamlit) - **Recommended for Beginners**

**Easiest way to use the system with a user-friendly interface.**

#### Quick Start:
```bash
# Option A: Using batch file (Windows)
run_web_interface.bat

# Option B: Manual command
streamlit run streamlit_app.py
```

#### Features:
- **📸 Image Upload Detection**: Upload photos for analysis
- **🎥 Live Webcam Detection**: Real-time processing
- **Interactive Controls**: Adjustable detection threshold
- **Visual Results**: Confidence bars and colored indicators
- **Detailed Metrics**: Comprehensive analysis results

#### How to Use:
1. **Run the application** using one of the methods above
2. **Open your browser** to the provided URL (usually `http://localhost:8501`)
3. **Choose a tab**:
   - **Image Upload**: Upload a photo containing a face
   - **Live Webcam**: Access real-time detection (requires webcam)
4. **Adjust settings** in the sidebar (detection threshold)
5. **View results** with confidence scores and visual feedback

### 2. 📹 Live Webcam Detection

**Real-time face anti-spoofing using your webcam.**

#### Quick Start:
```bash
# Option A: Using batch file (Windows)  
run_webcam_test.bat

# Option B: Manual command
python live_webcam_test.py
```

#### Advanced Usage:
```bash
# Custom threshold (0.0-1.0, higher = more strict)
python live_webcam_test.py --threshold 0.7

# Save video recording
python live_webcam_test.py --save_video output.avi

# Use different camera (0=default, 1=external, etc.)
python live_webcam_test.py --camera 1

# Different model
python live_webcam_test.py --model "saved_models/AntiSpoofing_bin_128.onnx"
```

#### Interactive Controls:
- **Q** or **ESC**: Quit the application
- **S**: Save current frame as image
- **R**: Reset statistics counter

#### Real-time Display:
- **Green Box**: Real face detected (✅ AUTHENTIC)
- **Red Box**: Fake face detected (❌ SPOOFING DETECTED)
- **Gray Box**: Uncertain result (⚠️ UNCERTAIN)
- **Live Statistics**: FPS, frame count, detection statistics
- **Confidence Bars**: Real-time confidence visualization

### 3. 🎬 Video Processing

**Process video files frame by frame with detection results.**

#### Quick Start:
```bash
# Option A: Using batch file (Windows)
run_video_test.bat

# Option B: Manual command
python video_test.py "path/to/video.mp4"
```

#### Advanced Usage:
```bash
# With live preview during processing
python video_test.py "input_video.mp4" --preview

# Custom output filename
python video_test.py "input.mp4" --output "custom_output.mp4"

# Custom threshold
python video_test.py "input.mp4" --threshold 0.6

# Different models
python video_test.py "input.mp4" --model "saved_models/AntiSpoofing_bin_128.onnx"
```

#### Output:
- **Processed video** with detection results overlaid
- **Frame-by-frame analysis** with bounding boxes
- **Statistics overlay** showing detection counts
- **Progress tracking** during processing
- **Final report** with comprehensive statistics

### 4. 📸 Static Image Testing

**Test individual images for anti-spoofing detection.**

#### Basic Usage:
```bash
# Test single image
python test_image.py "path/to/image.jpg"

# Test multiple images
python test_image.py "image1.jpg" "image2.jpg" "image3.jpg"

# Test sample images
python test_image.py "sample_test/Sample_photo_1.jpeg" "sample_test/Sample_photo_2.jpeg"
```

#### Advanced Usage:
```bash
# Save result images with bounding boxes
python test_image.py "image.jpg" --save_results

# Custom threshold
python test_image.py "image.jpg" --threshold 0.7

# Different model
python test_image.py "image.jpg" --model "saved_models/AntiSpoofing_bin_128.onnx"
```

#### Output:
- **Console results** with detailed confidence scores
- **Bounding box coordinates** for detected faces
- **Classification results** (Real/Fake/Uncertain)
- **Optional result images** with visual annotations
- **Batch processing summary** for multiple images

## 🔧 Command Line Options

### Global Parameters (Available for all scripts):

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--model` | Anti-spoofing model path | `saved_models/AntiSpoofing_bin_1.5_128.onnx` | `--model "models/custom.onnx"` |
| `--face_detector` | Face detection model path | `saved_models/yolov5s-face.onnx` | `--face_detector "models/yolo.onnx"` |
| `--threshold` | Detection threshold (0.0-1.0) | `0.5` | `--threshold 0.7` |

### Webcam-Specific Parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--camera` | Camera index | `0` | `--camera 1` |
| `--save_video` | Save video recording | `None` | `--save_video "recording.avi"` |

### Video-Specific Parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--output` | Output video path | `input_detected.mp4` | `--output "result.mp4"` |
| `--preview` | Show processing preview | `False` | `--preview` |

### Image-Specific Parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--save_results` | Save annotated images | `False` | `--save_results` |

## 🎯 Understanding Results

### Detection Classes:
- **✅ REAL FACE**: Live person detected (confidence > threshold)
- **❌ FAKE FACE**: Spoofing attempt detected (photo, video, mask)
- **⚠️ UNCERTAIN**: Low confidence result (needs manual review)
- **👤 NO FACE**: No face detected in image/frame

### Confidence Scores:
- **Real Confidence**: Probability that the face is from a live person (0.0-1.0)
- **Fake Confidence**: Probability that the face is a spoofing attempt (0.0-1.0)
- **Threshold**: Minimum real confidence required for "Real" classification

### Visual Indicators:
- **🟢 Green**: Real face detected
- **🔴 Red**: Fake face detected  
- **🟡 Yellow/Orange**: Uncertain result
- **🔵 Blue**: No face detected

## 🚀 Quick Start Examples

### Example 1: Test with Sample Data
```bash
# Test sample images
python test_image.py "sample_test/Sample_photo_1.jpeg" --save_results

# Test sample video
python video_test.py "sample_test/Video_Sample_1.mp4" --preview
```

### Example 2: Web Interface
```bash
# Start web interface
streamlit run streamlit_app.py
# Open browser to http://localhost:8501
```

### Example 3: Live Detection
```bash
# Basic webcam test
python live_webcam_test.py

# With recording
python live_webcam_test.py --save_video "my_test.avi" --threshold 0.6
```

## 🔧 Troubleshooting

### Common Issues:

#### 1. **Webcam Not Working**
```bash
# Check available cameras
python -c "import cv2; print('Camera 0:', cv2.VideoCapture(0).isOpened())"

# Try different camera index
python live_webcam_test.py --camera 1
```

#### 2. **Model Loading Errors**
- Ensure model files exist in `saved_models/` directory
- Check file paths in commands
- Verify ONNX runtime installation: `pip install onnxruntime`

#### 3. **Streamlit Issues**
```bash
# Install/update Streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

#### 4. **Performance Issues**
- Use CPU version for compatibility: `pip install onnxruntime` (not `onnxruntime-gpu`)
- Reduce video resolution or frame rate
- Close other applications using camera/CPU

### Hardware Requirements:
- **Minimum**: 4GB RAM, Intel i5 or equivalent
- **Recommended**: 8GB RAM, dedicated GPU (optional)
- **Webcam**: 720p or higher for best results

## 📊 Model Information

### Available Models:
1. **AntiSpoofing_bin_128.onnx**: Basic binary model (128x128 input)
2. **AntiSpoofing_bin_1.5_128.onnx**: Enhanced model with better accuracy
3. **yolov5s-face.onnx**: Face detection model

### Model Performance:
- **Input Size**: 128x128 pixels
- **Classification**: Binary (Real vs Fake)
- **Inference Speed**: ~10-50 FPS (depending on hardware)
- **Accuracy**: >95% on standard datasets

## 🛡️ Security Notes

- **Educational Purpose**: This system is designed for research and educational use
- **Production Use**: Additional validation recommended for security-critical applications
- **Privacy**: All processing is done locally - no data sent to external servers
- **Limitations**: Performance may vary with lighting conditions, camera quality, and face angles

## 📝 Batch Files (Windows)

For Windows users, convenient batch files are provided:

- **`run_web_interface.bat`**: Launch Streamlit web interface
- **`run_webcam_test.bat`**: Start live webcam detection
- **`run_video_test.bat`**: Process video files
- **`run_webcam_test_enhanced.bat`**: Enhanced webcam test with better model

## 🎓 Tips for Best Results

1. **Lighting**: Ensure good, even lighting on face
2. **Camera Position**: Face the camera directly
3. **Distance**: Stay 1-3 feet from camera
4. **Stability**: Keep camera steady for consistent results
5. **Threshold Tuning**: Adjust threshold based on your security needs
   - Higher threshold (0.7-0.9): More strict, fewer false positives
   - Lower threshold (0.3-0.5): More lenient, fewer false negatives

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Test with sample data first
4. Check camera/video file permissions

---

🔒 **Stay secure with AI-powered face anti-spoofing detection!**
