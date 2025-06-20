@echo off
echo Starting Face Anti-Spoofing Web Interface...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit
)

echo.
echo ========================================
echo   Face Anti-Spoofing Web Interface
echo ========================================
echo.
echo The web interface will open in your browser
echo Use Ctrl+C to stop the server
echo.

REM Run the Streamlit app
streamlit run streamlit_app.py

echo.
echo Web interface stopped. Press any key to exit.
pause
