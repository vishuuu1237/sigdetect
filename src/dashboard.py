"""Streamlit dashboard for SignalGuard pipeline.

Allows the user to upload an image, capture from webcam, or upload a video, runs the
SignalGuardPipeline on the input, displays the annotated result, and shows a table
of recent violations logged in `output/violations.csv`.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import sys
import os
from pathlib import Path

# Ensure src/ is on the import path so bare imports work (matches pipeline.py's own imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pipeline import SignalGuardPipeline

st.set_page_config(page_title="SignalGuard Dashboard", layout="wide")

st.title("🚦 SignalGuard Dashboard")

# Sidebar inputs
with st.sidebar:
    st.header("Input Media")
    input_type = st.radio("Select input type", ("Image", "Webcam", "Video"))
    uploaded_file = None
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    else:
        st.info("Webcam capture will be used when you press **Start**.")

# Initialise pipeline (reuse to avoid re‑loading models each frame)
pipeline = SignalGuardPipeline()

def read_image(file) -> np.ndarray:
    """Read uploaded image bytes into a BGR OpenCV array."""
    bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    return img

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Run the full pipeline on a single frame and return the annotated image."""
    result = pipeline.process_frame(frame)
    # Handle both tuple return (annotated, detections) and bare ndarray
    if isinstance(result, tuple):
        return result[0]
    return result

# Main area
if input_type == "Image" and uploaded_file:
    img = read_image(uploaded_file)
    annotated = process_frame(img)
    st.subheader("Annotated Image")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
elif input_type == "Video" and uploaded_file:
    # Save temporary file
    temp_path = Path("tmp_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    cap = cv2.VideoCapture(str(temp_path))
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_frame(frame)
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()
    temp_path.unlink(missing_ok=True)
elif input_type == "Webcam":
    img = st.camera_input("Take a picture")
    if img:
        frame = read_image(img)
        annotated = process_frame(frame)
        st.subheader("Annotated Capture")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
else:
    st.info("Select an input source from the sidebar to begin.")

# ---------------------------------------------------------------------
# Violations table & metrics
st.markdown("---")
st.header("Recent Violations")
csv_path = Path("output/violations.csv")
if csv_path.is_file():
    df = pd.read_csv(csv_path)
    # Convert timestamp to datetime for filtering
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S")
    today = datetime.datetime.now().date()
    df_today = df[df["timestamp_dt"].dt.date == today]
    # Metrics
    total_violations = len(df_today)
    # Placeholder for vehicles count – would require additional logging.
    total_vehicles = total_violations
    compliance_rate = 0.0 if total_vehicles == 0 else (1 - total_violations / total_vehicles) * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Vehicles Today", total_vehicles)
    col2.metric("Violations Today", total_violations)
    col3.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    # Show table (limit to last 20 entries)
    st.dataframe(df_today.drop(columns=["timestamp_dt"]).sort_values(by="timestamp", ascending=False).head(20))
    # Download button
    csv_bytes = df_today.to_csv(index=False).encode()
    st.download_button(
        label="Download Today's Violations CSV",
        data=csv_bytes,
        file_name="violations_today.csv",
        mime="text/csv",
    )
else:
    st.info("No violations logged yet.")
