"""
Live Webcam Anti-Spoofing Test Script
Tests the binary anti-spoofing model on live webcam feed
"""

import cv2
import numpy as np
import time
import argparse
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

# Color constants for visualization
COLOR_REAL = (0, 255, 0)      # Green for real faces
COLOR_FAKE = (0, 0, 255)      # Red for fake faces
COLOR_UNKNOWN = (127, 127, 127)  # Gray for uncertain
COLOR_NO_FACE = (255, 0, 0)   # Blue for no face detected

def increased_crop(img, bbox, bbox_inc=1.5):
    """
    Crop face region with increased bounding box for better model input
    """
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

def make_prediction(img, face_detector, anti_spoof):
    """
    Make anti-spoofing prediction on image
    Returns: bbox, label, confidence_score
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face
    bbox_result = face_detector([img_rgb])[0]
    
    if bbox_result.shape[0] > 0:
        bbox = bbox_result.flatten()[:4].astype(int)
    else:
        return None, None, None, "No face detected"

    # Crop face and make anti-spoofing prediction
    cropped_face = increased_crop(img_rgb, bbox, bbox_inc=1.5)
    pred = anti_spoof([cropped_face])[0]
    
    # For binary classification: [real_prob, fake_prob]
    real_confidence = pred[0][0]
    fake_confidence = pred[0][1] if len(pred[0]) > 1 else 1 - real_confidence
    
    # Determine label (0=Real, 1=Fake)
    label = np.argmax(pred[0])
    confidence = max(real_confidence, fake_confidence)
    
    return bbox, label, real_confidence, fake_confidence

def draw_results(frame, bbox, label, real_conf, fake_conf, threshold=0.5):
    """
    Draw bounding box and prediction results on frame
    """
    if bbox is None:
        # No face detected
        cv2.putText(frame, "NO FACE DETECTED", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NO_FACE, 2)
        return frame
    
    x1, y1, x2, y2 = bbox
    
    # Determine color and text based on prediction
    if label == 0:  # Real face
        if real_conf > threshold:
            text = f"REAL: {real_conf:.3f}"
            color = COLOR_REAL
            status = "AUTHENTIC"
        else:
            text = f"UNCERTAIN: {real_conf:.3f}"
            color = COLOR_UNKNOWN
            status = "UNCERTAIN"
    else:  # Fake face
        text = f"FAKE: {fake_conf:.3f}"
        color = COLOR_FAKE
        status = "SPOOFING DETECTED"
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw main label
    cv2.putText(frame, text, (x1, y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw status at top of frame
    cv2.putText(frame, status, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Draw confidence bars
    bar_width = 200
    bar_height = 20
    bar_x = 20
    bar_y = frame.shape[0] - 80
    
    # Real confidence bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * real_conf), bar_y + bar_height), 
                 COLOR_REAL, -1)
    cv2.putText(frame, f"Real: {real_conf:.3f}", (bar_x, bar_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Fake confidence bar
    bar_y += 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * fake_conf), bar_y + bar_height), 
                 COLOR_FAKE, -1)
    cv2.putText(frame, f"Fake: {fake_conf:.3f}", (bar_x, bar_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="Live Webcam Anti-Spoofing Test")
    parser.add_argument("--model", "-m", type=str, 
                       default="saved_models/AntiSpoofing_bin_1.5_128.onnx", 
                       help="Path to anti-spoofing ONNX model")
    parser.add_argument("--face_detector", "-f", type=str, 
                       default="saved_models/yolov5s-face.onnx", 
                       help="Path to face detector ONNX model")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                       help="Real face probability threshold (0.0-1.0)")
    parser.add_argument("--camera", "-c", type=int, default=0, 
                       help="Camera index (usually 0 for default webcam)")
    parser.add_argument("--save_video", "-s", type=str, default=None, 
                       help="Path to save output video (optional)")
    
    args = parser.parse_args()
    
    print("Loading models...")
    
    # Initialize models
    try:
        face_detector = YOLOv5(args.face_detector)
        anti_spoof = AntiSpoof(args.model)
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Initialize webcam
    print(f"Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
    
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"✓ Camera initialized: {frame_width}x{frame_height} @ {fps}fps")
    
    # Video writer for saving (optional)
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, 
                                     (frame_width, frame_height))
        print(f"✓ Saving video to: {args.save_video}")
    
    print("\n" + "="*50)
    print("LIVE ANTI-SPOOFING TEST")
    print("="*50)
    print("Instructions:")
    print("- Press 'q' or 'ESC' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'r' to reset statistics")
    print(f"- Threshold: {args.threshold}")
    print("="*50)
    
    # Statistics
    frame_count = 0
    real_count = 0
    fake_count = 0
    no_face_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Make prediction
            bbox, label, real_conf, fake_conf = make_prediction(frame, face_detector, anti_spoof)
            
            # Update statistics
            if bbox is None:
                no_face_count += 1
            elif label == 0 and real_conf > args.threshold:
                real_count += 1
            else:
                fake_count += 1
            
            # Draw results
            frame = draw_results(frame, bbox, label, real_conf, fake_conf, args.threshold)
            
            # Draw statistics
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            stats_text = [
                f"FPS: {avg_fps:.1f}",
                f"Frames: {frame_count}",
                f"Real: {real_count}",
                f"Fake: {fake_count}",
                f"No Face: {no_face_count}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (frame_width - 150, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Live Anti-Spoofing Test', frame)
            
            # Save video frame if requested
            if video_writer:
                video_writer.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save current frame
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):  # Reset statistics
                frame_count = 0
                real_count = 0
                fake_count = 0
                no_face_count = 0
                start_time = time.time()
                print("Statistics reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*30)
        print("SESSION SUMMARY")
        print("="*30)
        print(f"Total frames: {frame_count}")
        print(f"Real faces: {real_count}")
        print(f"Fake faces: {fake_count}")
        print(f"No face detected: {no_face_count}")
        print(f"Session duration: {elapsed_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print("="*30)

if __name__ == "__main__":
    main()
