import numpy as np


class Tracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.lost = {}

        self.max_lost = 15   # 🔥 increased → smoother

    def center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    def update(self, detections, frame):
        updated_tracks = {}
        updated_lost = {}

        # 🔥 match detections
        for det in detections:
            bbox = det["bbox"]
            c_det = self.center(bbox)

            best_id = None
            best_dist = 9999

            for track_id, track_box in self.tracks.items():
                c_track = self.center(track_box)
                dist = self.distance(c_det, c_track)

                if dist < 80 and dist < best_dist:  # 🔥 relaxed
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            updated_tracks[best_id] = bbox
            updated_lost[best_id] = 0

        # 🔥 KEEP LOST TRACKS (KEY FIX)
        for track_id in self.tracks:
            if track_id not in updated_tracks:
                lost_count = self.lost.get(track_id, 0) + 1

                if lost_count < self.max_lost:
                    updated_tracks[track_id] = self.tracks[track_id]
                    updated_lost[track_id] = lost_count

        self.tracks = updated_tracks
        self.lost = updated_lost

        # 🔥 RETURN ALL TRACKS (NOT ONLY DETECTED)
        results = []
        for track_id, bbox in self.tracks.items():
            results.append({
                "track_id": track_id,
                "bbox": bbox
            })

        return results