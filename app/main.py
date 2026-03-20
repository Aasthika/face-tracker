import os
import cv2
import time
from app.config_loader import ConfigLoader
from app.video_stream import VideoStream
from app.detector import FaceDetector
from app.pipeline import Pipeline


def process_video(video_path, config):
    print(f"\n🎥 Processing: {video_path}")

    config["video_path"] = video_path

    stream = VideoStream(config)
    detector = FaceDetector(config)
    pipeline = Pipeline(config)

    cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        frame = stream.read_frame()

        if frame is None:
            break

        frame = cv2.resize(frame, (640, 360))

        detections = detector.detect(frame)

        frame = pipeline.process(frame, detections)

        # 🔥 STABLE FPS
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev_time + 1e-6)
            prev_time = now

        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.imshow("Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    stream.release()


def main():
    print("🚀 Starting System...")

    config = ConfigLoader().get_all()
    input_folder = "input/"

    videos = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    for video in videos:
        path = os.path.join(input_folder, video)
        process_video(path, config)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()