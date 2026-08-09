"""
recognition/face_recognizer.py

InsightFace recognition optimized for macOS / Apple Silicon.
"""

import numpy as np

from insightface.app import FaceAnalysis

from config import settings
from utils.logger import get_logger


log = get_logger()


# ============================================================================
# GLOBAL MODEL
# ============================================================================

_face_app = None


# ============================================================================
# INITIALIZE INSIGHTFACE
# ============================================================================

def init_face_model():

    global _face_app

    log.info(
        "Loading InsightFace model pack: %s",
        settings.FACE_MODEL_NAME
    )

    log.info(
        "InsightFace providers: %s",
        settings.FACE_PROVIDERS
    )

    # --------------------------------------------------------
    # Create FaceAnalysis
    # --------------------------------------------------------

    _face_app = FaceAnalysis(

        name=settings.FACE_MODEL_NAME,

        providers=settings.FACE_PROVIDERS,

    )

    # --------------------------------------------------------
    # Prepare model
    # --------------------------------------------------------

    _face_app.prepare(

        ctx_id=0,

        det_size=settings.FACE_DET_SIZE

    )

    log.info(
        "InsightFace model initialized successfully."
    )

    return _face_app


# ============================================================================
# EXTRACT FACES
# ============================================================================

def extract_faces(frame):

    global _face_app

    # --------------------------------------------------------
    # Lazy initialization
    # --------------------------------------------------------

    if _face_app is None:

        init_face_model()

    # --------------------------------------------------------
    # Face detection + recognition
    # --------------------------------------------------------

    faces = _face_app.get(frame)

    output = []

    for face in faces:

        # ----------------------------------------------------
        # Normalized embedding
        # ----------------------------------------------------

        emb = getattr(
            face,
            "normed_embedding",
            None
        )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if emb is None:

            raw = face.embedding

            norm = np.linalg.norm(raw)

            if norm == 0:

                continue

            emb = raw / norm

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            face.bbox.tolist()
        )

        output.append({

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "embedding": np.asarray(
                emb,
                dtype=np.float32
            ),

        })

    return output


# ============================================================================
# NORMALIZE EMBEDDING
# ============================================================================

def normalize(embedding):

    emb = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(emb)

    if norm == 0:

        return emb

    return emb / norm


# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def cosine_similarities(
    embedding,
    gallery_matrix
):

    emb = normalize(
        embedding
    )

    gal = (
        gallery_matrix
        / np.maximum(
            np.linalg.norm(
                gallery_matrix,
                axis=1,
                keepdims=True
            ),
            1e-10
        )
    )

    return gal @ emb


# ============================================================================
# MATCH EMBEDDING
# ============================================================================

def match_embedding(
    embedding,
    gallery_matrix,
    threshold=None
):

    if (
        gallery_matrix is None
        or len(gallery_matrix) == 0
    ):

        return (
            -1,
            0.0,
            False
        )

    similarities = cosine_similarities(
        embedding,
        gallery_matrix
    )

    best = int(
        np.argmax(similarities)
    )

    best_similarity = float(
        similarities[best]
    )

    threshold_value = (
        threshold
        if threshold is not None
        else settings.FACE_MATCH_THRESHOLD
    )

    return (
        best,
        best_similarity,
        best_similarity >= threshold_value
    )


# ============================================================================
# AVERAGE EMBEDDINGS
# ============================================================================

def average_embedding(
    embeddings
):

    if not embeddings:

        raise ValueError(
            "average_embedding() needs "
            "at least one embedding"
        )

    mean = np.mean(
        np.vstack(embeddings),
        axis=0
    )

    return normalize(mean)