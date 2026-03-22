🚀 Intelligent Face Tracker System

📌 Overview

This project is a real-time intelligent face tracking system designed to detect, track, and recognize individuals from video or RTSP streams. It logs structured ENTRY and EXIT events, stores data in a database, captures images, and provides analytics through a dashboard.

⸻

🎯 Problem Statement

Build a real-time intelligent system that:
	•	Detects people from video/RTSP streams
	•	Tracks them across frames
	•	Recognizes unique identities
	•	Logs entry and exit events
	•	Stores structured data (database + logs + images)

⸻

🏗️ System Architecture

Pipeline Flow

Video / RTSP Stream
        ↓
YOLOv8 Detection
        ↓
Tracker (ID assignment)
        ↓
Face Recognition (InsightFace)
        ↓
Event Manager (ENTRY / EXIT)
        ↓
Logger (DB + logs + images)
        ↓
Dashboard + Heatmap


⸻

🧠 AI Planning

Approach

The system is designed as a modular AI pipeline combining computer vision, tracking, and database logging.

Key Components
	•	Detection → YOLOv8 for person detection
	•	Tracking → Custom centroid-based tracker
	•	Recognition → InsightFace embeddings with cosine similarity
	•	Event Management → Entry/Exit detection using timeout logic
	•	Logging → SQLite database + image storage + logs
	•	Visualization → Streamlit dashboard + heatmap

⸻

✨ Features
	•	✅ Real-time person detection (YOLOv8)
	•	✅ Multi-object tracking
	•	✅ Face recognition using embeddings
	•	✅ Unique visitor counting
	•	✅ Entry/Exit event detection
	•	✅ Image capture for each event
	•	✅ SQLite database storage
	•	✅ Log file generation
	•	✅ Heatmap visualization
	•	✅ Streamlit live dashboard
	•	✅ RTSP + Video file support

⸻

⚙️ Setup Instructions

1. Clone Repository

git clone https://github.com/Aasthika/face-tracker
cd face-tracker

2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run Main Application

python main.py

5. Run Dashboard

streamlit run dashboard/streamlit_app.py


⸻

⚙️ Configuration (config.json)

{
  "input_type": "video",
  "video_path": "input/video_sample1.mp4",
  "rtsp_url": "rtsp://username:password@ip:port/stream",
  "frame_skip": 3,
  "similarity_threshold": 0.68,
  "face_conf_threshold": 0.5,
  "recognition_cooldown": 5.0,
  "tracker": "bytetrack",
  "log_images": true,
  "use_gpu": false,
  "entry_exit_timeout": 10,
  "save_embeddings": true,
  "enable_dashboard": true,
  "enable_heatmap": true
}


⸻

📊 Compute Estimation

Component	        CPU Usage	        GPU Usage
YOLO Detection	    Medium	        High (if GPU enabled)
Face Recognition	Medium	        Medium
Tracking	        Low	            None
Logging	            Low	            None
Dashboard	        Low	            None


⸻

📂 Output Structure

logs/
  ├── events.log
  ├── entries/
  └── exits/

data/
  └── events.db

outputs/
  └── heatmaps/
        └── heatmap.png


⸻

📊 Database Schema

events table

Column	    Type
id	        INTEGER
person_id	INTEGER
event_type	TEXT
timestamp	TEXT
image_path	TEXT

unique_visitors table

Column	    Type
person_id	INTEGER
first_seen	TEXT


⸻

📸 Sample Outputs
	•	Event Logs → logs/events.log
	•	Captured Faces → logs/entries/, logs/exits/
	•	Database → data/events.db
	•	Heatmap → outputs/heatmaps/heatmap.png

⸻

⚠️ Assumptions
	•	Each person has at least one visible face
	•	Camera angle is relatively stable
	•	Lighting conditions are sufficient for detection
	•	No heavy occlusion in crowded scenes

⸻

🎥 Demo Video

👉 (Add your Loom or YouTube link here)

⸻

🚀 Future Improvements
	•	Improve tracking using DeepSORT / ByteTrack
	•	Add multi-camera support
	•	Deploy as cloud-based service
	•	Optimize GPU performance
	•	Add alert system (security use case)

⸻


This project is a part of a hackathon run by https://katomaran.com
:::