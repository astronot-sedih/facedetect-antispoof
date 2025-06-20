@echo off
echo Video Anti-Spoofing Test Script
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Available video files in current directory:
echo.
dir *.mp4 *.avi *.mov *.mkv 2>nul
echo.

REM Check if sample video exists
if exist "Video_Sample_1.mp4" (
    echo Found sample video: Video_Sample_1.mp4
    echo.
    
    set /p choice="Test with Video_Sample_1.mp4? (y/n): "
    if /i "%choice%"=="y" (
        echo.
        echo Testing with Video_Sample_1.mp4...
        echo Using enhanced model: AntiSpoofing_bin_1.5_128.onnx
        echo Output will be saved as: Video_Sample_1_detected.mp4
        echo.
        python video_test.py "Video_Sample_1.mp4" --preview
        goto :end
    )
)

REM Manual input
echo.
set /p video_path="Enter path to video file: "

if not exist "%video_path%" (
    echo Error: Video file not found: %video_path%
    pause
    exit /b 1
)

echo.
echo Processing video: %video_path%
echo Using enhanced model: AntiSpoofing_bin_1.5_128.onnx
echo.

python video_test.py "%video_path%" --preview

:end
echo.
echo Video processing completed!
echo Check the output file for results.
echo.
pause
