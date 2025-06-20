@echo off
echo Starting Live Webcam Anti-Spoofing Test...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the webcam test script
echo Running with default binary model (128x128)...
python live_webcam_test.py --model "saved_models/AntiSpoofing_bin_128.onnx" --threshold 0.5

echo.
echo Test completed. Press any key to exit.
pause
