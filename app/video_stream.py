import cv2
import time


class VideoStream:
    def __init__(self, config):
        self.input_type = config.get("input_type")
        self.video_path = config.get("video_path")
        self.rtsp_url = config.get("rtsp_url")

        self.cap = None
        self.last_frame_time = time.time()

        self.connect()

    def connect(self):
        """Initialize video source"""

        # 🔹 RTSP STREAM (MAIN INTERVIEW MODE)
        if self.input_type == "rtsp":
            print(f"📡 Connecting to RTSP: {self.rtsp_url}")

            # 🔥 IMPORTANT FIX (Mac + RTSP)
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise RuntimeError("❌ RTSP failed to open")

        # 🔹 VIDEO FILE (backup only)
        else:
            print(f"📂 Opening: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                raise RuntimeError("❌ Video file failed")

    def read_frame(self):
        """Read next frame"""

        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            print("⚠️ Stream lost → reconnecting...")
            self.reconnect()
            return None

        return frame

    def reconnect(self):
        """Reconnect RTSP"""

        if self.cap:
            self.cap.release()

        time.sleep(1)

        print("🔄 Reconnecting RTSP...")

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self.cap.isOpened():
            print("✅ Reconnected!")
        else:
            print("❌ Reconnect failed")

    def get_fps(self):
        now = time.time()
        fps = 1 / (now - self.last_frame_time + 1e-6)
        self.last_frame_time = now
        return int(fps)

    def release(self):
        if self.cap:
            self.cap.release()