#!/bin/bash
# Model Download Script
# This script downloads the required model files for the Face Anti-Spoofing Detection System

echo "📦 Downloading Face Anti-Spoofing Models..."
echo "================================================"

# Create saved_models directory if it doesn't exist
mkdir -p saved_models

echo "⏬ Downloading models..."
echo "Note: Replace these URLs with actual model download links"

# Placeholder URLs - Replace with actual model hosting URLs
echo "🤖 AntiSpoofing_bin_1.5_128.onnx (Enhanced Model)"
echo "🤖 AntiSpoofing_bin_128.onnx (Basic Model)" 
echo "👤 yolov5s-face.onnx (Face Detection Model)"

echo ""
echo "📋 Models needed in saved_models/ directory:"
echo "  - AntiSpoofing_bin_1.5_128.onnx  (~50MB)"
echo "  - AntiSpoofing_bin_128.onnx      (~30MB)"
echo "  - yolov5s-face.onnx             (~30MB)"
echo ""
echo "🔗 You can:"
echo "  1. Upload models to Google Drive and share public links"
echo "  2. Use GitHub Releases to attach model files"
echo "  3. Host models on Hugging Face Model Hub"
echo ""
echo "📝 Update this script with actual download URLs"
