import os
import cv2
from datetime import datetime
from app.database import Database


class Logger:
    def __init__(self, config):
        self.base_path = "logs"
        self.log_file = os.path.join(self.base_path, "events.log")

        os.makedirs(self.base_path, exist_ok=True)

        self.db = Database()

    def log_event(self, event_type, person_id, frame, bbox):
        print("🔥 LOGGER CALLED:", event_type, person_id)

        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")

        folder = os.path.join(
            self.base_path,
            "entries" if event_type == "ENTRY" else "exits",
            date_str
        )

        os.makedirs(folder, exist_ok=True)

        # SAFE CROP
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = frame[y1:y2, x1:x2]

        img_name = f"ID_{person_id}_{time_str}.jpg"
        img_path = os.path.join(folder, img_name)

        if face is not None and face.size != 0:
            cv2.imwrite(img_path, face)

        # LOG FILE
        log_line = f"[{datetime.now()}] {event_type} ID {person_id} → {img_path}\n"

        with open(self.log_file, "a") as f:
            f.write(log_line)

        # 🔥 FORCE DB WRITE WITH DEBUG
        try:
            print("👉 Writing to DB...")
            self.db.insert_event(person_id, event_type, img_path)
            print("✅ DB WRITE SUCCESS")
        except Exception as e:
            print("❌ DB ERROR:", e)