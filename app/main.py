import os
import cv2
import time

from app.config_loader import ConfigLoader
from app.video_stream import VideoStream
from app.detector import FaceDetector
from app.pipeline import Pipeline


def process_video(video_path, pipeline, config):
    print(f"\n🎥 Processing: {video_path}")
    config["video_path"] = video_path

    stream = VideoStream(config)

    pipeline.reset_events()

    cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0

    while True:
        frame = stream.read_frame()
        if frame is None:
            break

        frame = cv2.resize(frame, (640, 360))

        detections = detector.detect(frame)

        # 🔥 PROCESS FIRST (VERY IMPORTANT)
        frame = pipeline.process(frame, detections)

        # 🔥 FPS FIX (STABLE)
        current_time = time.time()
        fps = 1 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        # 🔥 UNIQUE COUNT AFTER PROCESS
        count = pipeline.logger.get_unique_count()

        # 🔥 DRAW FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # 🔥 DRAW UNIQUE
        cv2.putText(
            frame,
            f"Unique: {count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    stream.release()


def main():
    print("🚀 Starting Face Tracker System...")
    config = ConfigLoader().get_all()

    global detector
    detector = FaceDetector(config)
    pipeline = Pipeline(config)

    input_folder = "input/"

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"⚠️ Created '{input_folder}' — add videos and rerun.")
        return

    videos = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp4")])

    if not videos:
        print("⚠️ No .mp4 files found")
        return

    for video in videos:
        path = os.path.join(input_folder, video)
        process_video(path, pipeline, config)

    cv2.destroyAllWindows()

    print("\n✅ All videos processed.")

    # 🔥 SAVE HEATMAP
    pipeline.heatmap.save()


if __name__ == "__main__":
    main()