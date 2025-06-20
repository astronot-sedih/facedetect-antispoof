"""
Test Anti-Spoofing Model on Static Images
Tests the binary anti-spoofing model on provided image files
"""

import cv2
import numpy as np
import argparse
import os
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

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

def test_image(image_path, face_detector, anti_spoof, save_result=False):
    """
    Test anti-spoofing on a single image
    """
    print(f"\n📸 Testing image: {os.path.basename(image_path)}")
    print("-" * 50)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return None
    
    print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Convert to RGB for face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    print("🔍 Detecting faces...")
    bbox_result = face_detector([img_rgb])[0]
    
    if bbox_result.shape[0] == 0:
        print("❌ No faces detected in the image")
        return None
    
    # Get the first (largest) face
    bbox = bbox_result.flatten()[:4].astype(int)
    x1, y1, x2, y2 = bbox
    print(f"✅ Face detected at: ({x1}, {y1}) to ({x2}, {y2})")
    
    # Crop face for anti-spoofing
    print("✂️  Cropping face region...")
    cropped_face = increased_crop(img_rgb, bbox, bbox_inc=1.5)
    print(f"✅ Face cropped to: {cropped_face.shape[1]}x{cropped_face.shape[0]} pixels")
    
    # Make anti-spoofing prediction
    print("🤖 Running anti-spoofing prediction...")
    pred = anti_spoof([cropped_face])[0]
    
    # Extract probabilities
    real_confidence = pred[0][0]
    fake_confidence = pred[0][1] if len(pred[0]) > 1 else 1 - real_confidence
    
    # Determine label
    label = np.argmax(pred[0])
    
    # Results
    print("\n🔍 RESULTS:")
    print("=" * 30)
    print(f"Real Confidence: {real_confidence:.4f} ({real_confidence*100:.2f}%)")
    print(f"Fake Confidence: {fake_confidence:.4f} ({fake_confidence*100:.2f}%)")
    
    if label == 0:
        result = "REAL FACE" if real_confidence > 0.5 else "UNCERTAIN (Low confidence real)"
        color = "🟢" if real_confidence > 0.5 else "🟡"
    else:
        result = "FAKE FACE (Spoofing detected)"
        color = "🔴"
    
    print(f"Prediction: {color} {result}")
    print(f"Confidence: {max(real_confidence, fake_confidence):.4f}")
    
    # Save result image if requested
    if save_result:
        result_img = img.copy()
        
        # Draw bounding box
        bbox_color = (0, 255, 0) if label == 0 and real_confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), bbox_color, 3)
        
        # Add text
        text = f"{result} ({max(real_confidence, fake_confidence):.3f})"
        cv2.putText(result_img, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
        
        # Save
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"result_{base_name}.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"💾 Result saved as: {output_path}")
    
    return {
        'image_path': image_path,
        'bbox': bbox,
        'label': label,
        'real_confidence': real_confidence,
        'fake_confidence': fake_confidence,
        'result': result
    }

def main():
    parser = argparse.ArgumentParser(description="Test Anti-Spoofing Model on Images")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s) to test")
    parser.add_argument("--model", "-m", type=str, 
                       default="saved_models/AntiSpoofing_bin_1.5_128.onnx", 
                       help="Path to anti-spoofing ONNX model")
    parser.add_argument("--face_detector", "-f", type=str, 
                       default="saved_models/yolov5s-face.onnx", 
                       help="Path to face detector ONNX model")
    parser.add_argument("--save_results", "-s", action="store_true", 
                       help="Save result images with bounding boxes")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                       help="Real face probability threshold")
    
    args = parser.parse_args()
    
    print("🚀 Anti-Spoofing Image Test")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Face Detector: {args.face_detector}")
    print(f"Threshold: {args.threshold}")
    print(f"Images to test: {len(args.images)}")
    
    # Load models
    print("\n📦 Loading models...")
    try:
        face_detector = YOLOv5(args.face_detector)
        anti_spoof = AntiSpoof(args.model)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Test each image
    results = []
    for image_path in args.images:
        result = test_image(image_path, face_detector, anti_spoof, args.save_results)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print("\n📊 SUMMARY")
        print("=" * 50)
        real_count = sum(1 for r in results if r['label'] == 0 and r['real_confidence'] > args.threshold)
        fake_count = sum(1 for r in results if r['label'] == 1 or r['real_confidence'] <= args.threshold)
        
        print(f"Total images tested: {len(results)}")
        print(f"Real faces detected: {real_count}")
        print(f"Fake/Uncertain faces: {fake_count}")
        
        print("\nDetailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅ REAL" if result['label'] == 0 and result['real_confidence'] > args.threshold else "❌ FAKE/UNCERTAIN"
            print(f"{i}. {os.path.basename(result['image_path'])}: {status} (Real: {result['real_confidence']:.3f})")

if __name__ == "__main__":
    main()
