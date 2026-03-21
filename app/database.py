import sqlite3
import os
from datetime import datetime


class Database:
    def __init__(self, db_path="data/events.db"):
        os.makedirs("data", exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.create_tables()

    def create_tables(self):
        # EVENTS TABLE
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            event_type TEXT,
            timestamp TEXT,
            image_path TEXT
        )
        """)

        # 🔥 NEW: UNIQUE VISITORS TABLE
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS unique_visitors (
            person_id INTEGER PRIMARY KEY,
            first_seen TEXT
        )
        """)

        self.conn.commit()

    def insert_event(self, person_id, event_type, image_path):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("👉 INSERTING:", person_id, event_type)

        # INSERT EVENT
        self.cursor.execute("""
        INSERT INTO events (person_id, event_type, timestamp, image_path)
        VALUES (?, ?, ?, ?)
        """, (person_id, event_type, timestamp, image_path))

        # 🔥 UNIQUE VISITOR TRACKING
        if event_type == "ENTRY":
            self.cursor.execute("""
            INSERT OR IGNORE INTO unique_visitors (person_id, first_seen)
            VALUES (?, ?)
            """, (person_id, timestamp))

        self.conn.commit()

    # 🔥 GET UNIQUE COUNT
    def get_unique_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM unique_visitors")
        return self.cursor.fetchone()[0]