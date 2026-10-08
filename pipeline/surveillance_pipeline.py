"""
pipeline/surveillance_pipeline.py
---------------------------------
Real-time AI surveillance pipeline optimized for Apple Silicon (MPS / CoreML) and CUDA/CPU.

Key Capabilities:
    - YOLOv8 person detection with hardware acceleration
    - InsightFace ArcFace face extraction and cosine similarity recognition
    - Atomic latest-frame writer to prevent tearing
    - WebSocket callback support for ultra-low latency live rendering
    - Database-backed unknown person deduplication across restarts
    - Robust visit logging with configurable cooldowns
    - Graceful startup/shutdown control
"""

import os
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

# ---------------------------------------------------------------------------
# Global Pipeline State
# ---------------------------------------------------------------------------

_gallery = {
    "ids": [],
    "names": [],
    "roles": [],
    "matrix": None,
    "loaded_at": 0.0,
}

_last_visit_log = {}
_recent_unknowns = []

GALLERY_REFRESH_SEC = 30
LIVE_FRAME_FPS = getattr(settings, "LIVE_FRAME_FPS", 15)
LIVE_JPEG_QUALITY = getattr(settings, "LIVE_JPEG_QUALITY", 85)


def refresh_gallery(force=False):
    """Reload stakeholder gallery embeddings from SQLite."""
    now = time.time()
    if not force and _gallery["matrix"] is not None and (now - _gallery["loaded_at"] < GALLERY_REFRESH_SEC):
        return

    ids, names, roles, matrix = db_manager.load_stakeholder_gallery()
    _gallery["ids"] = ids
    _gallery["names"] = names
    _gallery["roles"] = roles
    _gallery["matrix"] = matrix
    _gallery["loaded_at"] = now

    if not ids:
        log.warning("Stakeholder recognition gallery is currently EMPTY. Register users to identify them.")
    else:
        log.info("Stakeholder gallery loaded: %d registered identities.", len(ids))


def _should_log_visit(stakeholder_id):
    """Check whether cooldown has elapsed for logging a visit."""
    now = time.time()
    last = _last_visit_log.get(stakeholder_id, 0.0)
    if now - last >= settings.VISIT_LOG_COOLDOWN_SEC:
        _last_visit_log[stakeholder_id] = now
        return True
    return False


def _init_unknown_cache():
    """Seed in-memory deduplication cache from recent database entries on startup."""
    global _recent_unknowns
    try:
        recent_db = db_manager.get_recent_unknown_embeddings(seconds=settings.UNKNOWN_LOG_COOLDOWN_SEC)
        _recent_unknowns = [(face_recognizer.normalize(emb), ts) for emb, ts in recent_db]
        log.info("Initialized unknown deduplication cache with %d recent records.", len(_recent_unknowns))
    except Exception as exc:
        log.warning("Could not warm-start unknown deduplication cache: %s", exc)
        _recent_unknowns = []


def _is_new_unknown(embedding):
    """Check if an unknown face embedding is distinct from recently logged unknowns."""
    now = time.time()

    # Expire old records outside cooldown window
    _recent_unknowns[:] = [
        (emb, ts) for emb, ts in _recent_unknowns
        if (now - ts < settings.UNKNOWN_LOG_COOLDOWN_SEC)
    ]

    normalized = face_recognizer.normalize(embedding)

    for prev_emb, _ts in _recent_unknowns:
        sim = float(np.dot(normalized, prev_emb))
        if sim >= settings.UNKNOWN_DUP_THRESHOLD:
            return False

    _recent_unknowns.append((normalized, now))
    return True


def _write_latest_frame(frame):
    """Atomically write latest annotated frame JPEG to disk without file locking / tearing."""
    if not settings.SAVE_LATEST_FRAME:
        return False

    final_path = settings.LATEST_FRAME_PATH
    temp_path = final_path + ".tmp.jpg"

    try:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        success = cv2.imwrite(
            temp_path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(LIVE_JPEG_QUALITY)]
        )
        if success:
            os.replace(temp_path, final_path)
            return True
        return False
    except Exception as exc:
        log.warning("Atomic frame write failed: %s", exc)
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return False


def process_frame(frame, camera_location):
    """
    Run YOLO person detector and InsightFace face recognizer on a single frame.
    Returns: (summary, persons, faces_for_display, events)
    """
    refresh_gallery()

    # 1. YOLO Person Detection
    persons = person_detector.detect_persons(frame)

    # 2. InsightFace Extraction
    detected_faces = face_recognizer.extract_faces(frame)

    summary = {
        "persons": len(persons),
        "recognized": 0,
        "unknown": 0,
    }

    faces_for_display = []
    events = []

    for face in detected_faces:
        # Match against gallery
        idx, similarity, is_match = face_recognizer.match_embedding(
            face["embedding"],
            _gallery["matrix"]
        )

        if is_match:
            summary["recognized"] += 1
            stakeholder_id = _gallery["ids"][idx]
            name = _gallery["names"][idx]
            role = _gallery["roles"][idx]
            label = f"{name} ({role}) {similarity:.2f}"

            faces_for_display.append({
                "box": face["box"],
                "label": label,
                "color": image_utils.COLOR_STAKEHOLDER,
            })

            if _should_log_visit(stakeholder_id):
                db_manager.log_visit(stakeholder_id, camera_location, similarity)
                events.append({
                    "type": "STAKEHOLDER_VISIT",
                    "name": name,
                    "role": role,
                    "location": camera_location,
                    "similarity": round(similarity, 3),
                    "timestamp": time.time()
                })
                log.info("VISIT LOGGED: %s (%s) @ %s [sim=%.3f]", name, role, camera_location, similarity)

        else:
            summary["unknown"] += 1
            label = f"UNKNOWN {similarity:.2f}"

            faces_for_display.append({
                "box": face["box"],
                "label": label,
                "color": image_utils.COLOR_UNKNOWN,
            })

            if _is_new_unknown(face["embedding"]):
                path = image_utils.save_face_crop(
                    frame,
                    face["box"],
                    settings.UNKNOWN_IMG_DIR,
                    "unknown"
                )
                if path is not None:
                    unk_id = db_manager.log_unknown(path, face["embedding"], camera_location)
                    events.append({
                        "type": "UNKNOWN_CAPTURED",
                        "id": unk_id,
                        "image_path": path,
                        "location": camera_location,
                        "timestamp": time.time()
                    })
                    log.warning("UNKNOWN RECORDED: #%s @ %s -> %s", unk_id, camera_location, path)

    return summary, persons, faces_for_display, events


def run_surveillance(
    source=None,
    camera_location=None,
    display=None,
    max_frames=None,
    frame_callback=None,
    stop_event=None,
):
    """
    Main surveillance loop.
    Supports optional `frame_callback(annotated_frame, summary, events)` for direct WebSocket broadcasting.
    Supports `stop_event` (threading.Event) for clean asynchronous termination.
    """
    settings.ensure_directories()
    db_manager.init_db()
    _init_unknown_cache()

    source = settings.DEFAULT_SOURCE if source is None else source
    camera_location = camera_location or settings.DEFAULT_CAMERA_LOCATION
    display = settings.DISPLAY_WINDOW if display is None else display

    log.info("Starting AI Models (YOLOv8 + InsightFace)...")
    person_detector.init_detector()
    face_recognizer.init_face_model()
    refresh_gallery(force=True)

    try:
        stream = webcam_stream.VideoStream(source)
    except Exception as exc:
        log.error("Could not initialize video stream: %s", exc)
        return

    log.info("Surveillance pipeline running. Source: %r, Location: %s", source, camera_location)

    frame_idx = 0
    processed_frames = 0
    failures = 0
    fps_start = time.time()
    fps_count = 0
    camera_fps = 0.0

    last_live_write = 0.0
    live_interval = 1.0 / max(LIVE_FRAME_FPS, 1.0)

    last_persons = []
    last_faces = []
    last_summary = {"persons": 0, "recognized": 0, "unknown": 0}

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                log.info("Stop event triggered. Terminating surveillance loop.")
                break

            ok, frame = stream.read()
            if not ok:
                failures += 1
                if failures >= 40:
                    log.error("Video stream unavailable after repeated attempts.")
                    break
                time.sleep(0.005)
                continue

            failures = 0
            frame_idx += 1
            fps_count += 1

            # Calculate FPS
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                camera_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Run AI periodically according to FRAME_PROCESS_EVERY_N
            current_events = []
            if frame_idx % settings.FRAME_PROCESS_EVERY_N == 0:
                try:
                    last_summary, last_persons, last_faces, current_events = process_frame(
                        frame, camera_location
                    )
                    processed_frames += 1
                except Exception as exc:
                    log.exception("Inference error on frame %d: %s", frame_idx, exc)

            # Draw AI overlays onto current frame for consistent rendering
            for det in last_persons:
                box = det.get("box")
                conf = float(det.get("conf", 0.0))
                if box:
                    image_utils.draw_box(frame, box, f"person {conf:.2f}", image_utils.COLOR_PERSON)

            for fdet in last_faces:
                box = fdet.get("box")
                lbl = fdet.get("label", "UNKNOWN")
                clr = fdet.get("color", image_utils.COLOR_UNKNOWN)
                if box:
                    image_utils.draw_box(frame, box, lbl, clr)

            # Draw HUD status bar
            hud_text = (
                f"{camera_location} | FPS: {camera_fps:.1f} | "
                f"Persons: {last_summary['persons']} | "
                f"Known: {last_summary['recognized']} | "
                f"Unknown: {last_summary['unknown']} | "
                f"AI: {settings.AI_DEVICE}"
            )
            image_utils.draw_header(frame, hud_text)

            # Write atomic live frame to disk
            now = time.time()
            if settings.SAVE_LATEST_FRAME and (now - last_live_write >= live_interval):
                if _write_latest_frame(frame):
                    last_live_write = now

            # Broadcast frame directly to WebSocket listeners if callback provided
            if frame_callback is not None:
                try:
                    frame_callback(frame, last_summary, current_events, camera_fps)
                except Exception as cb_exc:
                    log.debug("Frame callback failed: %s", cb_exc)

            # Local OpenCV window if enabled
            if display:
                cv2.imshow("Campus Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit requested from preview window.")
                    break

            if max_frames is not None and processed_frames >= max_frames:
                break

    except KeyboardInterrupt:
        log.info("Surveillance interrupted by user.")
    except Exception as exc:
        log.exception("Surveillance main loop failure: %s", exc)
    finally:
        try:
            stream.release()
        except Exception:
            pass

        if display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        log.info("Surveillance stopped. Processed %d AI frames (%d total frames).", processed_frames, frame_idx)