@echo off
echo Starting Live Webcam Anti-Spoofing Test (Enhanced Model)...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the webcam test script with enhanced model
echo Running with enhanced binary model (1.5_128)...
python live_webcam_test.py --model "saved_models/AntiSpoofing_bin_1.5_128.onnx" --threshold 0.6

echo.
echo Test completed. Press any key to exit.
pause
