"""
pipeline/surveillance_pipeline.py

"""

import time

import cv2
import numpy as np

from camera import webcam_stream
from config import settings
from database import db_manager
from detection import person_detector
from recognition import face_recognizer
from utils import image_utils
from utils.logger import get_logger

log = get_logger()

# Module-level runtime state
# ---------------------------------------------------------------------------
_gallery = {"ids": [], "names": [], "roles": [], "matrix": None, "loaded_at": 0.0}
_last_visit_log = {}      # stakeholder_id -> unix time of last DB log
_recent_unknowns = []     # list of (embedding, unix_time) for de-duplication

GALLERY_REFRESH_SEC = 60  # re-read stakeholder gallery from DB every minute


def refresh_gallery(force=False):
    """(Re)load the stakeholder gallery from the database."""
    now = time.time()
    if not force and _gallery["matrix"] is not None \
            and now - _gallery["loaded_at"] < GALLERY_REFRESH_SEC:
        return
    ids, names, roles, matrix = db_manager.load_stakeholder_gallery()
    _gallery.update(ids=ids, names=names, roles=roles,
                    matrix=matrix, loaded_at=now)
    if not ids:
        log.warning("Stakeholder gallery is EMPTY — every face will be "
                    "flagged UNKNOWN. Register people first: "
                    "python main.py register --uid S001 --name \"...\" "
                    "--role Student --webcam")
    else:
        log.info("Stakeholder gallery loaded: %d identities", len(ids))


def _should_log_visit(stakeholder_id):
    """Cooldown: avoid duplicate visit rows for someone standing in frame."""
    now = time.time()
    last = _last_visit_log.get(stakeholder_id, 0.0)
    if now - last >= settings.VISIT_LOG_COOLDOWN_SEC:
        _last_visit_log[stakeholder_id] = now
        return True
    return False


def _is_new_unknown(embedding):
    """
    De-duplicate unknowns: skip if a very similar face was saved recently.
    Also prunes expired cache entries.
    """
    now = time.time()
    _recent_unknowns[:] = [
        (emb, ts) for emb, ts in _recent_unknowns
        if now - ts < settings.UNKNOWN_LOG_COOLDOWN_SEC
    ]
    for emb, _ts in _recent_unknowns:
        if float(np.dot(face_recognizer.normalize(embedding), emb)) \
                >= settings.UNKNOWN_DUP_THRESHOLD:
            return False
    _recent_unknowns.append((face_recognizer.normalize(embedding), now))
    return True


def _face_inside_person(face_box, person_boxes):
    """True if the face-box centre lies inside any detected person box."""
    fx = (face_box[0] + face_box[2]) / 2.0
    fy = (face_box[1] + face_box[3]) / 2.0
    for pb in person_boxes:
        x1, y1, x2, y2 = pb["box"]
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            return True
    return False


# Per-frame processing
# ---------------------------------------------------------------------------
def process_frame(frame, camera_location):
    """
    Run detection + recognition + logging on one frame IN PLACE (annotations
    are drawn on `frame`). Returns a summary dict for the caller/tests.
    """
    refresh_gallery()

    persons = person_detector.detect_persons(frame)
    faces = face_recognizer.extract_faces(frame)

    summary = {"persons": len(persons), "recognized": 0, "unknown": 0}

    # Person boxes first (thin, informational)
    for det in persons:
        image_utils.draw_box(frame, det["box"],
                             f"person {det['conf']:.2f}",
                             image_utils.COLOR_PERSON)

    for face in faces:
        # If YOLO found people, only consider faces that belong to one of
        # them (suppresses posters/photos on walls at the frame edge).
        if persons and not _face_inside_person(face["box"], persons):
            continue

        idx, sim, is_match = face_recognizer.match_embedding(
            face["embedding"], _gallery["matrix"]
        )

        if is_match:
            summary["recognized"] += 1
            sid = _gallery["ids"][idx]
            name = _gallery["names"][idx]
            role = _gallery["roles"][idx]
            image_utils.draw_box(frame, face["box"],
                                 f"{name} ({role}) {sim:.2f}",
                                 image_utils.COLOR_STAKEHOLDER)
            if _should_log_visit(sid):
                db_manager.log_visit(sid, camera_location, sim)
                log.info("VISIT  | %s (%s) @ %s  sim=%.3f",
                         name, role, camera_location, sim)
        else:
            summary["unknown"] += 1
            image_utils.draw_box(frame, face["box"],
                                 f"UNKNOWN {sim:.2f}",
                                 image_utils.COLOR_UNKNOWN)
            if _is_new_unknown(face["embedding"]):
                path = image_utils.save_face_crop(
                    frame, face["box"], settings.UNKNOWN_IMG_DIR, "unknown")
                db_manager.log_unknown(path, face["embedding"], camera_location)
                log.warning("UNKNOWN person registered @ %s -> %s",
                            camera_location, path)

    return summary


# Main loop
# ---------------------------------------------------------------------------
def run_surveillance(source=None, camera_location=None, display=None,
                     max_frames=None):
    """
    Continuous surveillance loop.
    
    """
    settings.ensure_directories()
    db_manager.init_db()

    source = settings.DEFAULT_SOURCE if source is None else source
    camera_location = camera_location or settings.DEFAULT_CAMERA_LOCATION
    display = settings.DISPLAY_WINDOW if display is None else display

    # Warm the models up-front so the first frame isn't slow.
    person_detector.init_detector()
    face_recognizer.init_face_model()
    refresh_gallery(force=True)

    cap = webcam_stream.open_stream(source)
    if cap is None:
        return

    log.info("Surveillance started | source=%r location=%s", source,
             camera_location)

    frame_idx = processed = failures = 0
    fps_time, fps_count, fps = time.time(), 0, 0.0

    try:
        while True:
            ok, frame = webcam_stream.read_frame(cap)
            if not ok:
                failures += 1
                if failures >= 30:          # ~stream ended / network dead
                    log.error("Stream ended or unreachable — stopping.")
                    break
                time.sleep(0.1)
                continue
            failures = 0
            frame_idx += 1

            # Frame skipping for real-time throughput
            if frame_idx % settings.FRAME_PROCESS_EVERY_N != 0:
                continue

            process_frame(frame, camera_location)
            processed += 1

            # FPS measurement (processing rate)
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count / (time.time() - fps_time)
                fps_time, fps_count = time.time(), 0

            image_utils.draw_header(
                frame, f"{camera_location} | {fps:.1f} FPS | q = quit")

            if settings.SAVE_LATEST_FRAME:
                cv2.imwrite(settings.LATEST_FRAME_PATH, frame)

            if display:
                cv2.imshow("Campus Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and processed >= max_frames:
                break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        webcam_stream.release_stream(cap)
        if display:
            cv2.destroyAllWindows()
        log.info("Surveillance stopped | frames processed=%d", processed)
