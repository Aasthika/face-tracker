import time


class EventManager:
    def __init__(self, config):
        self.active_tracks = {}   # track_id → last_seen
        self.logged_ids = set()   # unique persons entered
        self.timeout = config.get("entry_exit_timeout", 10)

    def update(self, tracks, track_to_id):
        current_time = time.time()
        events = []

        current_track_ids = set()

        for track in tracks:
            track_id = track["track_id"]
            person_id = track_to_id.get(track_id)

            if not isinstance(person_id, int):
                continue

            current_track_ids.add(track_id)

            # ✅ STRICT ENTRY (NO DUPLICATES)
            if person_id not in self.logged_ids:
                events.append({
                    "type": "ENTRY",
                    "id": person_id,
                    "bbox": track["bbox"]   # 🔥 store bbox
                })
                self.logged_ids.add(person_id)

            self.active_tracks[track_id] = {
                "time": current_time,
                "id": person_id,
                "bbox": track["bbox"]   # 🔥 store last bbox
            }

        # 🔥 EXIT FIX (IMPORTANT)
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_track_ids:

                data = self.active_tracks[track_id]

                if current_time - data["time"] > self.timeout:
                    events.append({
                        "type": "EXIT",
                        "id": data["id"],
                        "bbox": data["bbox"]   # 🔥 USE STORED BBOX
                    })

                    del self.active_tracks[track_id]

        return events