import insightface
import numpy as np


class FaceRecognizer:
    def __init__(self, config):
        self.similarity_threshold = config.get("similarity_threshold", 0.6)

        self.app = insightface.app.FaceAnalysis(
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        print("✅ Recognizer ready")

        self.embeddings = []
        self.ids = []
        self.next_id = 1

    def get_embedding(self, face):
        if face is None or face.size == 0:
            return None

        faces = self.app.get(face)
        if len(faces) == 0:
            return None

        return faces[0].embedding

    def match(self, emb):
        if not self.embeddings:
            return None

        sims = [
            np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e))
            for e in self.embeddings
        ]

        idx = int(np.argmax(sims))
        if sims[idx] > self.similarity_threshold:
            return self.ids[idx]

        return None

    def register(self, emb):
        self.embeddings.append(emb)
        self.ids.append(self.next_id)
        self.next_id += 1
        return self.ids[-1]