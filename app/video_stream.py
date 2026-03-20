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
        """Initialize video source with fallback"""

        # 🔹 Try RTSP first
        if self.input_type == "rtsp":
            print(f"📡 Connecting to RTSP: {self.rtsp_url}")

            self.cap = cv2.VideoCapture(self.rtsp_url)

            if not self.cap.isOpened():
                print("⚠️ RTSP failed → switching to video file")
                self.input_type = "video"
                self.cap = cv2.VideoCapture(self.video_path)

        # 🔹 Video mode
        else:
            print(f"📂 Opening: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)

        # ❌ Final check
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("❌ Failed to open video source")

    def read_frame(self):
        """Read next frame"""

        ret, frame = self.cap.read()

        if not ret:
            # 🎬 Video finished
            if self.input_type == "video":
                print("✅ Video finished")
                return None

            # 📡 RTSP lost → reconnect
            elif self.input_type == "rtsp":
                print("⚠️ RTSP lost → reconnecting...")
                self.reconnect()
                return None

        return frame

    def reconnect(self):
        """Reconnect RTSP stream"""

        if self.cap:
            self.cap.release()

        time.sleep(2)

        self.cap = cv2.VideoCapture(self.rtsp_url)

        if self.cap.isOpened():
            print("✅ Reconnected successfully!")
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
