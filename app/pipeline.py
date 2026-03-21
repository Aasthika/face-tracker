import cv2
import time
import numpy as np


class Pipeline:
    def __init__(self, config):
        from app.recognizer import FaceRecognizer
        from app.tracker import Tracker
        from app.event_manager import EventManager
        from app.logger import Logger

        from app.heatmap import HeatmapGenerator
        self.heatmap = HeatmapGenerator()

        self.recognizer = FaceRecognizer(config)
        self.tracker = Tracker()
        self.event_manager = EventManager(config)
        self.logger = Logger(config)

        self.config = config

        self.track_to_id = {}
        self.last_recognition = {}
        self.cooldown = config.get("recognition_cooldown", 3.0)
        self.face_conf_threshold = config.get("face_conf_threshold", 0.5)

    # ─────────────────────────────────────────

    def process(self, frame, detections):
        current_time = time.time()

        tracks = self.tracker.update(detections, frame)
        self.heatmap.update(frame, tracks)

        for track in tracks:
            track_id = track["track_id"]
            x1, y1, x2, y2 = track["bbox"]

            # 🔥 RECOGNITION (FIXED)
            self._try_recognize(frame, track_id, x1, y1, x2, y2, current_time)

            # 🔥 ALWAYS DRAW
            self._draw(frame, track_id, x1, y1, x2, y2)

        # EVENTS
        events = self.event_manager.update(tracks, self.track_to_id)

        for e in events:
            print(f"{e['type']} → ID {e['id']}")
            self.logger.log_event(e["type"], e["id"], frame, e["bbox"])

        return frame

    # ─────────────────────────────────────────

    def _try_recognize(self, frame, track_id, x1, y1, x2, y2, current_time):

        # already assigned → skip
        if track_id in self.track_to_id:
            return

        last = self.last_recognition.get(track_id, 0)
        if current_time - last < self.cooldown:
            return

        self.last_recognition[track_id] = current_time

        # crop person (upper part)
        h_frame, w_frame = frame.shape[:2]

        px1 = max(0, x1)
        py1 = max(0, y1)
        px2 = min(w_frame, x2)
        py2 = min(h_frame, y1 + int((y2 - y1) * 0.6))

        person_crop = frame[py1:py2, px1:px2]

        if person_crop is None or person_crop.size == 0:
            return

        h, w = person_crop.shape[:2]
        if h < 40 or w < 30:
            return

        # 🔥 FIX: SINGLE FACE DETECTION ONLY
        faces = self.recognizer.app.get(person_crop)

        if len(faces) == 0:
            return

        # pick best face
        face_obj = max(faces, key=lambda f: f.det_score)

        if face_obj.det_score < self.face_conf_threshold:
            return

        # 🔥 DIRECT EMBEDDING (NO RE-DETECTION)
        emb = face_obj.embedding
        emb = emb / np.linalg.norm(emb)

        # MATCH OR REGISTER
        match = self.recognizer.match(emb)

        if match is not None:
            self.track_to_id[track_id] = match
        else:
            new_id = self.recognizer.register(emb)
            self.track_to_id[track_id] = new_id

    # ─────────────────────────────────────────

    def _draw(self, frame, track_id, x1, y1, x2, y2):
        person_id = self.track_to_id.get(track_id)
        lost = self.tracker.lost.get(track_id, 0)

        color = (0, 255, 0) if lost == 0 else (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if person_id is not None:
            label = f"ID {person_id} | T{track_id}"
        else:
            label = f"T{track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # ─────────────────────────────────────────

    def reset_events(self):
        from app.event_manager import EventManager
        from app.tracker import Tracker

        self.event_manager = EventManager(self.config)
        self.tracker = Tracker()
        self.track_to_id = {}
        self.last_recognition = {}