"""
recognition/face_recognizer.py
"""

import numpy as np
from insightface.app import FaceAnalysis

from config import settings
from utils.logger import get_logger

log = get_logger()

_face_app = None  # module-level InsightFace handle


def init_face_model():
    """Load the InsightFace analysis pack (detector + ArcFace embedder)."""
    global _face_app
    log.info("Loading InsightFace model pack: %s", settings.FACE_MODEL_NAME)
    _face_app = FaceAnalysis(
        name=settings.FACE_MODEL_NAME,
        providers=settings.FACE_PROVIDERS,
    )
    # ctx_id=0 -> first GPU when CUDA provider is used, otherwise CPU.
    _face_app.prepare(ctx_id=0, det_size=settings.FACE_DET_SIZE)
    return _face_app


def extract_faces(frame):
    """
    Detect faces in a BGR frame and return a list of dicts.
    """
    global _face_app
    if _face_app is None:
        init_face_model()

    faces = _face_app.get(frame)
    output = []
    for face in faces:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:  # fall back: normalize the raw embedding ourselves
            raw = face.embedding
            norm = np.linalg.norm(raw)
            if norm == 0:
                continue
            emb = raw / norm
        x1, y1, x2, y2 = face.bbox.tolist()
        output.append({
            "box": (x1, y1, x2, y2),
            "embedding": np.asarray(emb, dtype=np.float32),
        })
    return output


def normalize(embedding):
    """L2-normalize an embedding vector (no-op for zero vectors)."""
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb if norm == 0 else emb / norm


def cosine_similarities(embedding, gallery_matrix):
    """
    Cosine similarity of one embedding vs. an (N, D) gallery matrix.
    """
    emb = normalize(embedding)
    gal = gallery_matrix / np.maximum(
        np.linalg.norm(gallery_matrix, axis=1, keepdims=True), 1e-10
    )
    return gal @ emb


def match_embedding(embedding, gallery_matrix, threshold=None):
    """
    Match one embedding against the gallery.
    """
    if gallery_matrix is None or len(gallery_matrix) == 0:
        return -1, 0.0, False

    sims = cosine_similarities(embedding, gallery_matrix)
    best = int(np.argmax(sims))
    best_sim = float(sims[best])
    thresh = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD
    return best, best_sim, best_sim >= thresh


def average_embedding(embeddings):
    """
    Average several embeddings of the SAME person into one robust template
    (mean, then re-normalized). Used at registration time.
    """
    if not embeddings:
        raise ValueError("average_embedding() needs at least one embedding")
    mean = np.mean(np.vstack(embeddings), axis=0)
    return normalize(mean)
