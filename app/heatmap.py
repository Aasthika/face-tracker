import numpy as np
import cv2
import os


class HeatmapGenerator:
    def __init__(self):
        self.heatmap = None

    def update(self, frame, tracks):
        h, w = frame.shape[:2]

        if self.heatmap is None:
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if 0 <= cx < w and 0 <= cy < h:
                self.heatmap[cy, cx] += 1

    def save(self):
        if self.heatmap is None:
            return

        heat = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        heat = heat.astype(np.uint8)

        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        os.makedirs("outputs/heatmaps", exist_ok=True)
        cv2.imwrite("outputs/heatmaps/heatmap.png", heat_color)

        print("🔥 Heatmap saved → outputs/heatmaps/heatmap.png")