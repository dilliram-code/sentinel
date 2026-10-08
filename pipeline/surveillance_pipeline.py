"""
pipeline/surveillance_pipeline.py
---------------------------------
Real-time AI surveillance pipeline optimized for Apple Silicon (MPS / CoreML) and CUDA/CPU.

Key Capabilities:
    - Decoupled asynchronous architecture for buttery smooth 30+ FPS video streaming
    - YOLOv8 person detection with hardware acceleration
    - InsightFace ArcFace face extraction and cosine similarity recognition
    - Non-blocking latest-frame file writer to prevent tearing
    - WebSocket callback support for ultra-low latency live rendering
    - Database-backed unknown person deduplication across restarts
    - Robust visit logging with configurable cooldowns
    - Graceful startup/shutdown control
"""

import os
import time
import threading
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
# Global Pipeline State & Gallery Cache
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
LIVE_FRAME_FPS = getattr(settings, "LIVE_FRAME_FPS", 30)
LIVE_JPEG_QUALITY = getattr(settings, "LIVE_JPEG_QUALITY", 80)


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
    """Safely write latest annotated frame JPEG to disk without file locking / tearing."""
    if not settings.SAVE_LATEST_FRAME:
        return False

    final_path = settings.LATEST_FRAME_PATH
    temp_path = f"{final_path}.{os.getpid()}_{threading.get_ident()}.tmp.jpg"

    try:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        success = cv2.imwrite(
            temp_path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(LIVE_JPEG_QUALITY)]
        )
        if success and os.path.exists(temp_path):
            os.replace(temp_path, final_path)
            return True
        return False
    except Exception as exc:
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


# ---------------------------------------------------------------------------
# Asynchronous Decoupled AI Processor
# ---------------------------------------------------------------------------

class AsyncAIProcessor:
    """
    Dedicated background worker thread for AI inference (YOLO + InsightFace).
    Prevents deep learning inference from blocking camera frame acquisition,
    allowing the video stream to render at full native 30+ FPS.
    """

    def __init__(self, camera_location):
        self.camera_location = camera_location
        self.pending_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.results_lock = threading.Lock()
        self.running = True

        self.last_persons = []
        self.last_faces = []
        self.last_summary = {"persons": 0, "recognized": 0, "unknown": 0}
        self.last_events = []
        self.ai_fps = 0.0
        self.ai_inference_time_ms = 0.0
        self.total_inferences = 0

        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncAIWorker")
        self.thread.start()

    def submit_frame(self, frame):
        """Submit a frame for AI processing without blocking the streaming loop."""
        with self.frame_lock:
            self.pending_frame = frame.copy()
            self.new_frame_event.set()

    def get_latest_results(self):
        """Retrieve the freshest inference results instantaneously (<0.01ms)."""
        with self.results_lock:
            events = self.last_events
            self.last_events = []  # Clear consumed events
            return (
                self.last_summary.copy(),
                list(self.last_persons),
                list(self.last_faces),
                events,
                self.ai_fps,
                self.ai_inference_time_ms,
                self.total_inferences
            )

    def _worker_loop(self):
        fps_start = time.time()
        fps_count = 0

        while self.running:
            self.new_frame_event.wait(timeout=0.05)
            if not self.running:
                break
            if not self.new_frame_event.is_set():
                continue

            frame_to_process = None
            with self.frame_lock:
                frame_to_process = self.pending_frame
                self.pending_frame = None
                self.new_frame_event.clear()

            if frame_to_process is None:
                continue

            t0 = time.perf_counter()
            try:
                summary, persons, faces, events = process_frame(
                    frame_to_process, self.camera_location
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                fps_count += 1
                self.total_inferences += 1

                elapsed = time.time() - fps_start
                ai_fps = self.ai_fps
                if elapsed >= 1.0:
                    ai_fps = fps_count / elapsed
                    fps_count = 0
                    fps_start = time.time()

                with self.results_lock:
                    self.last_summary = summary
                    self.last_persons = persons
                    self.last_faces = faces
                    if events:
                        self.last_events.extend(events)
                    self.ai_fps = ai_fps
                    self.ai_inference_time_ms = dt_ms

            except Exception as exc:
                log.exception("Async AI inference error: %s", exc)

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Main Surveillance Runner
# ---------------------------------------------------------------------------

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
    Executes decoupled multi-threading:
        - Stream capture & rendering runs at native camera rate (30+ FPS)
        - AI models run asynchronously in background thread
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

    ai_processor = AsyncAIProcessor(camera_location=camera_location)

    frame_idx = 0
    failures = 0
    fps_start = time.time()
    fps_count = 0
    camera_fps = 0.0

    last_live_write = 0.0
    live_interval = 1.0 / max(LIVE_FRAME_FPS, 1.0)

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                log.info("Stop event triggered. Terminating surveillance loop.")
                break

            ok, frame = stream.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 40:
                    log.error("Video stream unavailable after repeated attempts.")
                    break
                time.sleep(0.005)
                continue

            failures = 0
            frame_idx += 1
            fps_count += 1

            # Calculate actual live video streaming FPS
            now = time.time()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                camera_fps = fps_count / elapsed
                fps_count = 0
                fps_start = now

            # Non-blocking dispatch to async AI worker
            if frame_idx % max(1, settings.FRAME_PROCESS_EVERY_N) == 0:
                ai_processor.submit_frame(frame)

            # Instantaneous fetch of latest AI recognition annotations
            summary, persons, faces, events, ai_fps, ai_time_ms, total_inf = ai_processor.get_latest_results()

            # Render detection boxes on current frame for smooth overlay
            for det in persons:
                box = det.get("box")
                conf = float(det.get("conf", 0.0))
                if box:
                    image_utils.draw_box(frame, box, f"person {conf:.2f}", image_utils.COLOR_PERSON)

            for fdet in faces:
                box = fdet.get("box")
                lbl = fdet.get("label", "UNKNOWN")
                clr = fdet.get("color", image_utils.COLOR_UNKNOWN)
                if box:
                    image_utils.draw_box(frame, box, lbl, clr)

            # Draw HUD status bar
            hud_text = (
                f"{camera_location} | Stream: {camera_fps:.1f} FPS | "
                f"AI: {ai_fps:.1f} inf/s ({ai_time_ms:.0f}ms) | "
                f"Persons: {summary['persons']} | "
                f"Identified: {summary['recognized']} | "
                f"Unknown: {summary['unknown']}"
            )
            image_utils.draw_header(frame, hud_text)

            # Write latest frame to disk periodically without blocking
            if settings.SAVE_LATEST_FRAME and (now - last_live_write >= live_interval):
                if _write_latest_frame(frame):
                    last_live_write = now

            # Broadcast frame directly to WebSocket listeners
            if frame_callback is not None:
                try:
                    frame_callback(frame, summary, events, camera_fps)
                except Exception as cb_exc:
                    log.debug("Frame callback failed: %s", cb_exc)

            # Local OpenCV preview window if enabled
            if display:
                cv2.imshow("Campus Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit requested from preview window.")
                    break

            if max_frames is not None and (total_inf >= max_frames or frame_idx >= max_frames * 5):
                break

    except KeyboardInterrupt:
        log.info("Surveillance interrupted by user.")
    except Exception as exc:
        log.exception("Surveillance main loop failure: %s", exc)
    finally:
        ai_processor.stop()

        try:
            stream.release()
        except Exception:
            pass

        if display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        log.info("Surveillance stopped. Processed %d AI inferences (%d video frames rendered).", ai_processor.total_inferences, frame_idx)