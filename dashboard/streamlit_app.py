import sys
import os

# ✅ FIX IMPORT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import sqlite3
import time
import pandas as pd
import psutil

from app.config_loader import ConfigLoader
from app.video_stream import VideoStream
from app.detector import FaceDetector
from app.pipeline import Pipeline

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(BASE_DIR, "data", "events.db")
HEATMAP_PATH = os.path.join(BASE_DIR, "outputs", "heatmaps", "heatmap.png")
LOG_FILE = os.path.join(BASE_DIR, "logs", "events.log")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
INPUT_DIR = os.path.join(BASE_DIR, "input")

# ===============================
# UI
# ===============================
st.set_page_config(layout="wide")
st.title("🚀 Intelligent Face Tracker Dashboard")

# ===============================
# STATE
# ===============================
if "running" not in st.session_state:
    st.session_state.running = False

if "mode" not in st.session_state:
    st.session_state.mode = "Video"

if "video_index" not in st.session_state:
    st.session_state.video_index = 0

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ===============================
# CONTROLS
# ===============================
col1, col2, col3 = st.columns(3)

if col1.button("▶️ Start"):
    st.session_state.running = True

if col2.button("⏹ Stop"):
    st.session_state.running = False

mode = col3.selectbox("Mode", ["Video", "RTSP"])
st.session_state.mode = mode

# ===============================
# CONFIG
# ===============================
config = ConfigLoader().get_all()

# ===============================
# DB FUNCTIONS
# ===============================
def get_unique():
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    val = conn.execute("SELECT COUNT(*) FROM unique_visitors").fetchone()[0]
    conn.close()
    return val

def get_total():
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    val = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    return val

def get_df():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM events", conn)
    conn.close()
    return df

# ===============================
# INIT SYSTEM
# ===============================
if st.session_state.running and not st.session_state.initialized:

    st.session_state.detector = FaceDetector(config)
    st.session_state.pipeline = Pipeline(config)

    if st.session_state.mode == "Video":
        videos = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")])
        st.session_state.videos = videos
        st.session_state.video_index = 0

    st.session_state.initialized = True

# ===============================
# LIVE FEED (NO FLICKER)
# ===============================
st.markdown("## 🎥 Live Feed")

frame_placeholder = st.empty()
metric_placeholder = st.empty()

if st.session_state.running:

    detector = st.session_state.detector
    pipeline = st.session_state.pipeline

    # ================= RTSP MODE =================
    if st.session_state.mode == "RTSP":

        rtsp_url = st.text_input("Enter RTSP URL", config.get("rtsp_url", ""))

        if not rtsp_url:
            st.warning("Enter RTSP URL")
            st.stop()

        stream = VideoStream({
            **config,
            "input_type": "rtsp",
            "rtsp_url": rtsp_url
        })

        prev_time = time.time()

        while st.session_state.running:

            frame = stream.read_frame()

            if frame is None:
                st.warning("⚠️ RTSP disconnected")
                break

            frame = cv2.resize(frame, (640, 360))

            detections = detector.detect(frame)
            frame = pipeline.process(frame, detections)

            # FPS
            now = time.time()
            fps = int(1 / (now - prev_time + 1e-6))
            prev_time = now

            unique = get_unique()

            cv2.putText(frame, f"FPS: {fps}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, f"Unique: {unique}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame, channels="RGB")
            metric_placeholder.metric("👥 Live Visitors", unique)

            time.sleep(0.03)

    # ================= VIDEO MODE =================
    else:

        videos = st.session_state.videos

        while st.session_state.running:

            idx = st.session_state.video_index

            if idx >= len(videos):
                st.success("✅ All videos processed")
                pipeline.heatmap.save()
                st.session_state.running = False
                break

            video_path = os.path.join(INPUT_DIR, videos[idx])

            st.write(f"🎬 Now Playing: {videos[idx]} ({idx+1}/{len(videos)})")

            stream = VideoStream({
                **config,
                "video_path": video_path,
                "input_type": "video"
            })

            prev_time = time.time()

            while st.session_state.running:

                frame = stream.read_frame()

                if frame is None:
                    stream.release()
                    st.session_state.video_index += 1
                    break

                frame = cv2.resize(frame, (640, 360))

                detections = detector.detect(frame)
                frame = pipeline.process(frame, detections)

                # FPS
                now = time.time()
                fps = int(1 / (now - prev_time + 1e-6))
                prev_time = now

                unique = get_unique()

                cv2.putText(frame, f"FPS: {fps}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"Unique: {unique}", (20,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(frame, channels="RGB")
                metric_placeholder.metric("👥 Live Visitors", unique)

                time.sleep(0.03)

# ===============================
# DASHBOARD
# ===============================
st.markdown("## 📊 Overview")

col1, col2 = st.columns(2)

col1.metric("👥 Unique Visitors", get_unique())
col2.metric("📌 Total Events", get_total())

# ===============================
# PERFORMANCE
# ===============================
st.markdown("## ⚡ Performance")

c1, c2 = st.columns(2)
c1.metric("CPU", f"{psutil.cpu_percent()}%")
c2.metric("RAM", f"{psutil.virtual_memory().percent}%")

# ===============================
# HEATMAP
# ===============================
st.markdown("## 🔥 Heatmap")

if os.path.exists(HEATMAP_PATH):
    st.image(HEATMAP_PATH, width="stretch")
else:
    st.warning("Heatmap will appear after processing")

# ===============================
# LOGS 
# ===============================
st.markdown("## 📜 Logs (Full History)")

if os.path.exists(LOG_FILE):

    with open(LOG_FILE, "r") as f:
        logs = f.readlines()

    if logs:
        # 🔥 Latest first
        logs = logs[::-1]

        # 🔥 Scrollable container (VERY IMPORTANT)
        log_text = "".join(logs)

        st.text_area(
            "All Events",
            log_text,
            height=300
        )
    else:
        st.info("No logs found")

else:
    st.warning("Log file not found")

# ===============================
# IMAGES (SHOW ALL LOG IMAGES)
# ===============================
st.markdown("## 📸 All Captured Faces")

def get_all_images(base_folder):
    all_images = []

    if not os.path.exists(base_folder):
        return []

    # 🔥 Walk through ALL subfolders (IMPORTANT)
    for root, dirs, files in os.walk(base_folder):
        for f in files:
            if f.endswith(".jpg"):
                full_path = os.path.join(root, f)
                all_images.append(full_path)

    # 🔥 Sort latest first
    all_images = sorted(all_images, key=os.path.getmtime, reverse=True)

    return all_images


# 🔥 GET ALL IMAGES
entry_images = get_all_images(os.path.join(LOGS_DIR, "entries"))
exit_images = get_all_images(os.path.join(LOGS_DIR, "exits"))


# ===============================
# DISPLAY ENTRIES
# ===============================
st.subheader("🟢 Entry Images")

if entry_images:
    cols = st.columns(6)
    for i, img in enumerate(entry_images[:60]):
        cols[i % 6].image(img, width="stretch")
else:
    st.info("No entry images found")


# ===============================
# DISPLAY EXITS
# ===============================
st.subheader("🔴 Exit Images")

if exit_images:
    cols = st.columns(6)
    for i, img in enumerate(exit_images[:60]):
        cols[i % 6].image(img, width="stretch")
else:
    st.info("No exit images found")