"""
Video Anti-Spoofing Test Script
Tests the binary anti-spoofing model on video files
Processes video frame by frame and saves output with detection results
"""

import cv2
import numpy as np
import argparse
import os
import time
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
    Make anti-spoofing prediction on image frame
    Returns: bbox, label, real_confidence, fake_confidence
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face
    bbox_result = face_detector([img_rgb])[0]
    
    if bbox_result.shape[0] > 0:
        bbox = bbox_result.flatten()[:4].astype(int)
    else:
        return None, None, None, None

    # Crop face and make anti-spoofing prediction
    cropped_face = increased_crop(img_rgb, bbox, bbox_inc=1.5)
    pred = anti_spoof([cropped_face])[0]
    
    # For binary classification: [real_prob, fake_prob]
    real_confidence = pred[0][0]
    fake_confidence = pred[0][1] if len(pred[0]) > 1 else 1 - real_confidence
    
    # Determine label (0=Real, 1=Fake)
    label = np.argmax(pred[0])
    
    return bbox, label, real_confidence, fake_confidence

def draw_results(frame, bbox, label, real_conf, fake_conf, threshold=0.5, frame_number=0, stats=None):
    """
    Draw bounding box and prediction results on frame
    """
    # Frame info
    cv2.putText(frame, f"Frame: {frame_number}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if bbox is None:
        # No face detected
        cv2.putText(frame, "NO FACE DETECTED", (20, 70), 
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
    cv2.putText(frame, status, (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Draw confidence information
    cv2.putText(frame, f"Real: {real_conf:.3f} | Fake: {fake_conf:.3f}", 
               (20, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw statistics if provided
    if stats:
        stats_text = [
            f"Total Frames: {stats['total']}",
            f"Real: {stats['real']} | Fake: {stats['fake']}",
            f"No Face: {stats['no_face']} | FPS: {stats['fps']:.1f}"
        ]
        
        for i, stat_text in enumerate(stats_text):
            cv2.putText(frame, stat_text, (frame.shape[1] - 300, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def process_video(input_path, output_path, face_detector, anti_spoof, threshold=0.5, show_preview=False):
    """
    Process video file frame by frame
    """
    print(f"📹 Processing video: {os.path.basename(input_path)}")
    print("-" * 50)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {input_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"✅ Video loaded: {frame_width}x{frame_height} @ {fps:.2f}fps")
    print(f"📊 Total frames: {total_frames} | Duration: {duration:.1f}s")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
      # Statistics
    stats = {
        'total': 0,
        'real': 0,
        'fake': 0,
        'no_face': 0,
        'fps': 0.0
    }
    
    # Processing loop
    start_time = time.time()
    frame_number = 0
    
    print("\n🚀 Starting video processing...")
    print("Press 'q' to quit early if preview is enabled")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            stats['total'] = frame_number
            
            # Make prediction
            bbox, label, real_conf, fake_conf = make_prediction(frame, face_detector, anti_spoof)
              # Update statistics
            if bbox is None:
                stats['no_face'] += 1
            elif label == 0 and real_conf is not None and real_conf > threshold:
                stats['real'] += 1
            else:
                stats['fake'] += 1
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            stats['fps'] = frame_number / elapsed_time if elapsed_time > 0 else 0
            
            # Draw results on frame
            result_frame = draw_results(frame, bbox, label, real_conf, fake_conf, 
                                      threshold, frame_number, stats)
            
            # Write frame to output video
            out.write(result_frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('Video Processing Preview', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Processing interrupted by user")
                    break
            
            # Progress update
            if frame_number % 30 == 0:  # Every 30 frames
                progress = (frame_number / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% | Frame {frame_number}/{total_frames} | FPS: {stats['fps']:.1f}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*50)
    print("VIDEO PROCESSING COMPLETE")
    print("="*50)
    print(f"📁 Output saved: {output_path}")
    print(f"🎬 Processed frames: {stats['total']}")
    print(f"✅ Real faces detected: {stats['real']}")
    print(f"❌ Fake faces detected: {stats['fake']}")
    print(f"👤 No face detected: {stats['no_face']}")
    print(f"⏱️ Processing time: {elapsed_time:.1f}s")
    print(f"🚀 Average FPS: {stats['fps']:.1f}")
    print("="*50)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Video Anti-Spoofing Test")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Path to output video file (default: adds '_detected' to input name)")
    parser.add_argument("--model", "-m", type=str, 
                       default="saved_models/AntiSpoofing_bin_1.5_128.onnx", 
                       help="Path to anti-spoofing ONNX model")
    parser.add_argument("--face_detector", "-f", type=str, 
                       default="saved_models/yolov5s-face.onnx", 
                       help="Path to face detector ONNX model")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                       help="Real face probability threshold (0.0-1.0)")
    parser.add_argument("--preview", "-p", action="store_true", 
                       help="Show live preview during processing")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input video file not found: {args.input}")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        extension = os.path.splitext(args.input)[1]
        args.output = f"{base_name}_detected{extension}"
    
    print("🚀 Video Anti-Spoofing Detection")
    print("=" * 50)
    print(f"📹 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print(f"🤖 Model: {args.model}")
    print(f"👤 Face Detector: {args.face_detector}")
    print(f"🎯 Threshold: {args.threshold}")
    print(f"👁️ Preview: {'Enabled' if args.preview else 'Disabled'}")
    
    # Load models
    print("\n📦 Loading models...")
    try:
        face_detector = YOLOv5(args.face_detector)
        anti_spoof = AntiSpoof(args.model)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Process video
    stats = process_video(
        args.input, 
        args.output, 
        face_detector, 
        anti_spoof, 
        args.threshold, 
        args.preview
    )
    
    if stats:
        print(f"\n🎉 Video processing completed successfully!")
        print(f"📁 Check output file: {args.output}")

if __name__ == "__main__":
    main()
