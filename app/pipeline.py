import cv2
import time


class Pipeline:
    def __init__(self, config):
        from app.recognizer import FaceRecognizer
        from app.tracker import Tracker
        from app.event_manager import EventManager
        from app.logger import Logger

        self.recognizer = FaceRecognizer(config)
        self.tracker = Tracker()
        self.event_manager = EventManager(config)
        self.logger = Logger(config)

        self.track_to_id = {}
        self.last_run = 0
        self.cooldown = 0.5

    def process(self, frame, detections):
        current_time = time.time()

        tracks = self.tracker.update(detections, frame)

        for track in tracks:
            track_id = track["track_id"]
            x1, y1, x2, y2 = track["bbox"]

            person_id = self.track_to_id.get(track_id, "...")

            # 🔥 RECOGNITION
            if current_time - self.last_run > self.cooldown:

                face_y2 = y1 + int((y2 - y1) * 0.5)
                face = frame[y1:face_y2, x1:x2]

                if face is not None and face.size != 0:
                    emb = self.recognizer.get_embedding(face)

                    if emb is not None:
                        match = self.recognizer.match(emb)

                        if match:
                            person_id = match
                        else:
                            if track_id not in self.track_to_id:
                                person_id = self.recognizer.register(emb)

                        self.track_to_id[track_id] = person_id

                self.last_run = current_time

            # 🔥 DRAW
            lost_frames = self.tracker.lost.get(track_id, 0)
            color = (0, 255, 0) if lost_frames == 0 else (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"T{track_id} | ID {person_id}" if person_id != "..." else f"T{track_id}"

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        # 🔥 EVENTS (UPDATED)
        events = self.event_manager.update(tracks, self.track_to_id)

        for e in events:
            print(f"{e['type']} → ID {e['id']}")

            # 🔥 DIRECT USE BBOX FROM EVENT (FIX)
            self.logger.log_event(
                e["type"],
                e["id"],
                frame,
                e["bbox"]
            )

        return frame