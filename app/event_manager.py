import time


class EventManager:
    def __init__(self, config):
        self.timeout = config.get("entry_exit_timeout", 10)

        # track_id → {"time": float, "id": int, "bbox": list}
        self.active_tracks = {}

        # person_id → currently inside (True) or not
        # Use set of person_ids currently "inside" the scene
        self.active_persons = set()

    def update(self, tracks, track_to_id):
        """
        Call once per frame with current tracks and the track→person_id map.
        Returns list of event dicts: {"type": "ENTRY"/"EXIT", "id": int, "bbox": list}
        """
        current_time = time.time()
        events = []
        current_track_ids = set()

        # ── Process currently visible tracks ────────────────────────────────
        for track in tracks:
            track_id = track["track_id"]
            person_id = track_to_id.get(track_id)

            # Only fire events for confirmed identities (int IDs)
            if not isinstance(person_id, int):
                continue

            current_track_ids.add(track_id)

            # Update last-seen info for this track
            self.active_tracks[track_id] = {
                "time": current_time,
                "id": person_id,
                "bbox": track["bbox"]
            }

            # ✅ ENTRY: fire only if person is NOT currently active
            if person_id not in self.active_persons:
                events.append({
                    "type": "ENTRY",
                    "id": person_id,
                    "bbox": track["bbox"]
                })
                self.active_persons.add(person_id)

        # ── Check for exits (tracks gone longer than timeout) ────────────────
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_track_ids:
                data = self.active_tracks[track_id]
                elapsed = current_time - data["time"]

                if elapsed > self.timeout:
                    person_id = data["id"]
                    events.append({
                        "type": "EXIT",
                        "id": person_id,
                        "bbox": data["bbox"]
                    })
                    # ✅ Remove from active so re-entry fires ENTRY again
                    self.active_persons.discard(person_id)
                    del self.active_tracks[track_id]

        return events