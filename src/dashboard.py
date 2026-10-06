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

st.set_page_config(page_title="SignalGuard — AI Traffic Compliance", page_icon="🚦", layout="wide")

st.title("🚦 SignalGuard — AI Traffic & Compliance Monitor")

# Sidebar settings & inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    conf_thresh = st.slider("Detection Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    
    st.markdown("---")
    st.header("📹 Input Source")
    input_type = st.radio("Select input type", ("Image", "Webcam", "Video"))
    uploaded_file = None
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload a traffic image", type=["png", "jpg", "jpeg"])
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
    else:
        st.info("Webcam capture will be used below.")

@st.cache_resource
def get_pipeline():
    """Cache pipeline instance across Streamlit reruns."""
    return SignalGuardPipeline()

pipeline = get_pipeline()

def read_image(file) -> np.ndarray:
    """Read uploaded image bytes into a BGR OpenCV array."""
    bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    return img

# Main processing area
if input_type == "Image" and uploaded_file:
    img = read_image(uploaded_file)
    with st.spinner("Processing image and running OCR..."):
        annotated, detections = pipeline.process_frame(img, conf_threshold=conf_thresh)
    
    col_img, col_stats = st.columns([2, 1])
    with col_img:
        st.subheader("Annotated Detection")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col_stats:
        st.subheader("Frame Insights")
        total_veh = len(detections)
        violations_count = sum(1 for d in detections if not d["compliant"])
        plates_found = sum(1 for d in detections if d.get("plate"))

        st.metric("Vehicles Detected", total_veh)
        st.metric("Number Plates Read", plates_found)
        st.metric("Violations Flagged", violations_count)

        if detections:
            st.markdown("##### Detected Entities")
            for idx, d in enumerate(detections):
                badge_style = "🟢 Compliant" if d["compliant"] else "🔴 Violation"
                with st.expander(f"Vehicle #{idx+1}: {d['type'].title()} ({d['color'].title()}) — {badge_style}", expanded=True):
                    st.write(f"**License Plate:** `{d['plate'] if d['plate'] else 'None detected'}`")
                    st.write(f"**Occupants:** {d['occupants']}")
                    if not d["compliant"]:
                        st.error(f"**Reason:** {d['violations'][0] if d['violations'] else 'Non-compliant'}")

elif input_type == "Video" and uploaded_file:
    temp_path = Path("tmp_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    cap = cv2.VideoCapture(str(temp_path))
    stframe = st.empty()
    stop_button = st.button("Stop Video Playback")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _ = pipeline.process_frame(frame, conf_threshold=conf_thresh)
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    cap.release()
    temp_path.unlink(missing_ok=True)

elif input_type == "Webcam":
    img = st.camera_input("Take a photo")
    if img:
        frame = read_image(img)
        with st.spinner("Analyzing capture..."):
            annotated, detections = pipeline.process_frame(frame, conf_threshold=conf_thresh)
        st.subheader("Annotated Capture")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    st.info("👈 Upload an image or video from the sidebar to begin traffic compliance analysis.")

# ---------------------------------------------------------------------
# Violations table & metrics
st.markdown("---")
st.header("📋 Recent Violations Log")
csv_path = Path("output/violations.csv")
if csv_path.is_file():
    try:
        df = pd.read_csv(csv_path)
        if not df.empty and "timestamp" in df.columns:
            st.dataframe(df.sort_values(by="timestamp", ascending=False).head(25), use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Violations Log (CSV)",
                data=csv_bytes,
                file_name="violations_log.csv",
                mime="text/csv",
            )
        else:
            st.info("No violations logged yet.")
    except Exception as e:
        st.info(f"Log file exists but cannot be parsed: {e}")
else:
    st.info("No violations logged yet.")

