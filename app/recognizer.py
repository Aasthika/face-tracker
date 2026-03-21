import insightface
import numpy as np
import pickle
import os


class FaceRecognizer:
    def __init__(self, config):
        # 🔥 FIX: lower threshold (was too strict)
        self.similarity_threshold = 0.6

        self.save_embeddings = config.get("save_embeddings", True)
        self.embeddings_path = "data/embeddings.pkl"

        self.app = insightface.app.FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if config.get("use_gpu", False)
            else ["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0 if config.get("use_gpu", False) else -1,
            det_size=(320, 320)
        )

        print("✅ Recognizer ready")

        self.embeddings = []
        self.ids = []
        self.next_id = 1

        if self.save_embeddings:
            self._load_embeddings()

    # ─────────────────────────────────────────
    # 🔥 FIXED: NO DOUBLE DETECTION
    # ─────────────────────────────────────────
    def get_embedding_from_face(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        try:
            faces = self.app.get(face_img)
        except Exception:
            return None

        if not faces:
            return None

        emb = faces[0].embedding

        norm = np.linalg.norm(emb)
        if norm == 0:
            return None

        return emb / norm

    # ─────────────────────────────────────────

    def match(self, emb):
        if not self.embeddings:
            return None

        sims = [float(np.dot(emb, e)) for e in self.embeddings]
        idx = int(np.argmax(sims))

        if sims[idx] > self.similarity_threshold:
            return self.ids[idx]

        return None

    def register(self, emb):
        self.embeddings.append(emb)
        self.ids.append(self.next_id)

        person_id = self.next_id
        self.next_id += 1

        if self.save_embeddings:
            self._save_embeddings()

        print(f"🆕 New Person → ID {person_id}")
        return person_id

    # ─────────────────────────────────────────

    def _save_embeddings(self):
        os.makedirs("data", exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "ids": self.ids,
                "next_id": self.next_id
            }, f)

    def _load_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            return

        try:
            with open(self.embeddings_path, "rb") as f:
                data = pickle.load(f)

            self.embeddings = data.get("embeddings", [])
            self.ids = data.get("ids", [])
            self.next_id = data.get("next_id", 1)

            print(f"📂 Loaded {len(self.ids)} identities")

        except Exception as e:
            print(f"⚠️ Load error: {e}")