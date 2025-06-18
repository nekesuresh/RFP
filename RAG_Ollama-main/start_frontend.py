#!/usr/bin/env python3
"""
Startup script for the Multi-Agent RFP Assistant Streamlit frontend
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("🎨 Starting Multi-Agent RFP Assistant Frontend...")
    print("📍 UI will be available at: http://localhost:8501")
    print("🔗 Make sure the backend is running at http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_ui/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]) 