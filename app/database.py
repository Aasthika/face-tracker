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
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            event_type TEXT,
            timestamp TEXT,
            image_path TEXT
        )
        """)
        self.conn.commit()

    def insert_event(self, person_id, event_type, image_path):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("👉 INSERTING:", person_id, event_type)

        self.cursor.execute("""
        INSERT INTO events (person_id, event_type, timestamp, image_path)
        VALUES (?, ?, ?, ?)
        """, (person_id, event_type, timestamp, image_path))

        self.conn.commit()