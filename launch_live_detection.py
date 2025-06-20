"""
Live Detection Launcher
Helper script to launch the live webcam detection from Streamlit
"""

import subprocess
import sys
import os
import argparse

def launch_live_detection(threshold=0.5, save_video=None, camera=0):
    """Launch the live webcam detection script"""
    
    # Build command
    cmd = [sys.executable, "live_webcam_test.py"]
    cmd.extend(["--threshold", str(threshold)])
    cmd.extend(["--camera", str(camera)])
    
    if save_video:
        cmd.extend(["--save_video", save_video])
    
    print(f"🚀 Launching: {' '.join(cmd)}")
    
    try:
        # Launch in a new process
        if os.name == 'nt':  # Windows
            # Use CREATE_NEW_CONSOLE to open in new window
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.getcwd()
            )
        else:  # Linux/Mac
            process = subprocess.Popen(cmd)
        
        print(f"✅ Live detection launched with PID: {process.pid}")
        return process.pid
        
    except Exception as e:
        print(f"❌ Error launching live detection: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Live Webcam Detection")
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    parser.add_argument("--save_video", "-s", type=str, default=None)
    parser.add_argument("--camera", "-c", type=int, default=0)
    
    args = parser.parse_args()
    
    launch_live_detection(
        threshold=args.threshold,
        save_video=args.save_video,
        camera=args.camera
    )
