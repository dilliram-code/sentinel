"""
registration/register_stakeholder.py
------------------------------------
Enroll stakeholders into the recognition gallery.

Two entry points:
    register_from_images(uid, name, role, image_paths)  -> from photo files
    register_from_webcam(uid, name, role, ...)          -> capture live samples

Multiple images per person are averaged into one robust ArcFace template.
"""

import glob
import os
import shutil
import time

import cv2

from config import settings
from database import db_manager
from recognition import face_recognizer
from utils.logger import get_logger

log = get_logger()

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _collect_image_paths(path):
    """Accept a single file, a directory, or a glob pattern."""
    if os.path.isdir(path):
        files = []
        for ext in VALID_EXT:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        return sorted(set(files))
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(path))


def _embedding_from_image(image_path):
    """Return the embedding of the LARGEST face in an image, or None."""
    frame = cv2.imread(image_path)
    if frame is None:
        log.warning("Unreadable image skipped: %s", image_path)
        return None
    faces = face_recognizer.extract_faces(frame)
    if not faces:
        log.warning("No face found in: %s", image_path)
        return None
    largest = max(
        faces,
        key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
    )
    return largest["embedding"]


def _store(uid, name, role, embeddings, reference_image=None):
    """Average embeddings, archive a reference photo, write to the DB."""
    template = face_recognizer.average_embedding(embeddings)

    saved_ref = None
    if reference_image and os.path.isfile(reference_image):
        os.makedirs(settings.STAKEHOLDER_IMG_DIR, exist_ok=True)
        ext = os.path.splitext(reference_image)[1] or ".jpg"
        saved_ref = os.path.join(settings.STAKEHOLDER_IMG_DIR, f"{uid}{ext}")
        shutil.copyfile(reference_image, saved_ref)

    sid = db_manager.add_stakeholder(uid, name, role, template, saved_ref)
    log.info("Registered stakeholder #%d | %s (%s) uid=%s from %d image(s)",
             sid, name, role, uid, len(embeddings))
    return sid


def register_from_images(uid, name, role, images_path):
    """
    Enroll from photo(s). `images_path` = file, folder, or glob pattern.
    Returns the stakeholder id, or None if no usable face was found.
    """
    settings.ensure_directories()
    db_manager.init_db()
    face_recognizer.init_face_model()

    paths = _collect_image_paths(images_path)
    if not paths:
        log.error("No images found at: %s", images_path)
        return None

    embeddings, reference = [], None
    for p in paths:
        emb = _embedding_from_image(p)
        if emb is not None:
            embeddings.append(emb)
            reference = reference or p

    if not embeddings:
        log.error("No detectable face in any provided image — not registered.")
        return None
    return _store(uid, name, role, embeddings, reference)


def register_from_webcam(uid, name, role, camera_index=0, samples=5,
                         delay_sec=1.0):
    """
    Capture `samples` face snapshots from a webcam (one every `delay_sec`)
    and enroll the person. Press 'q' to abort. Returns stakeholder id or None.
    """
    settings.ensure_directories()
    db_manager.init_db()
    face_recognizer.init_face_model()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open webcam index %s", camera_index)
        return None

    embeddings, reference_path, last_capture = [], None, 0.0
    log.info("Webcam enrollment for %s — look at the camera (%d samples)...",
             name, samples)
    try:
        while len(embeddings) < samples:
            ok, frame = cap.read()
            if not ok:
                continue

            preview = frame.copy()
            cv2.putText(preview,
                        f"Enrolling {name}: {len(embeddings)}/{samples} (q=quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.imshow("Stakeholder Enrollment", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Enrollment aborted by user.")
                return None

            if time.time() - last_capture < delay_sec:
                continue
            faces = face_recognizer.extract_faces(frame)
            if not faces:
                continue
            largest = max(
                faces,
                key=lambda f: (f["box"][2] - f["box"][0]) *
                              (f["box"][3] - f["box"][1]),
            )
            embeddings.append(largest["embedding"])
            last_capture = time.time()

            if reference_path is None:      # keep the first snapshot as photo
                os.makedirs(settings.STAKEHOLDER_IMG_DIR, exist_ok=True)
                reference_path = os.path.join(
                    settings.STAKEHOLDER_IMG_DIR, f"{uid}.jpg")
                cv2.imwrite(reference_path, frame)
            log.info("Captured sample %d/%d", len(embeddings), samples)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not embeddings:
        log.error("No samples captured — not registered.")
        return None

    template = face_recognizer.average_embedding(embeddings)
    sid = db_manager.add_stakeholder(uid, name, role, template, reference_path)
    log.info("Registered stakeholder #%d | %s (%s) uid=%s via webcam",
             sid, name, role, uid)
    return sid
