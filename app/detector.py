from ultralytics import YOLO


class FaceDetector:
    def __init__(self, config):
        # 🔥 Better model (important for crowd)
        self.model = YOLO("yolov8s.pt")
        self.model.fuse()

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # 🔥 LOWER threshold → detect far people
            if cls == 0 and conf > 0.15:
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                })

        return detections