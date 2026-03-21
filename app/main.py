import os
import cv2
import time

from app.config_loader import ConfigLoader
from app.video_stream import VideoStream
from app.detector import FaceDetector
from app.pipeline import Pipeline


def process_video(video_path, pipeline, config):
    """
    Process one video file using a SHARED pipeline.
    Only events are reset between videos — recognizer memory is kept.
    """
    print(f"\n🎥 Processing: {video_path}")
    config["video_path"] = video_path

    stream = VideoStream(config)

    # ✅ Reset tracker + event state only (NOT recognizer)
    pipeline.reset_events()

    cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        frame = stream.read_frame()
        if frame is None:
            break

        frame = cv2.resize(frame, (640, 360))   # already doing this ✅ — keep it
        detections = detector.detect(frame)
        frame = pipeline.process(frame, detections)

        # FPS display (updated every 10 frames)
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev_time + 1e-6)
            prev_time = now

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imshow("Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

    stream.release()


def main():
    print("🚀 Starting Face Tracker System...")
    config = ConfigLoader().get_all()

    # ✅ Create pipeline ONCE — recognizer memory persists across all videos
    global detector
    detector = FaceDetector(config)
    pipeline = Pipeline(config)

    input_folder = "input/"
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"⚠️  Created '{input_folder}' — add .mp4 files and rerun.")
        return

    videos = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp4")])
    if not videos:
        print("⚠️  No .mp4 files found in input/")
        return

    for video in videos:
        path = os.path.join(input_folder, video)
        process_video(path, pipeline, config)

    cv2.destroyAllWindows()
    print("\n✅ All videos processed.")


if __name__ == "__main__":
    main()