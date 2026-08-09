"""
pipeline/surveillance_pipeline.py

Real-time campus surveillance pipeline optimized for Apple Silicon.

Features:
    - YOLOv8 person detection
    - Apple MPS acceleration for YOLO
    - InsightFace recognition
    - CoreML acceleration when available
    - Background camera capture
    - Latest-frame-only processing
    - Persistent detection overlays
    - Atomic latest-frame JPEG writing
    - Reduced Streamlit flickering
    - Unknown-person registration
    - Visit logging
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


# ============================================================================
# RUNTIME STATE
# ============================================================================

_gallery = {
    "ids": [],
    "names": [],
    "roles": [],
    "matrix": None,
    "loaded_at": 0.0,
}


_last_visit_log = {}

_recent_unknowns = []


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

GALLERY_REFRESH_SEC = 60

# How often the annotated frame is written to disk.
#
# The camera can run at ~30 FPS while the dashboard only needs
# around 10-15 FPS.
LIVE_FRAME_FPS = getattr(
    settings,
    "LIVE_FRAME_FPS",
    12
)

LIVE_JPEG_QUALITY = getattr(
    settings,
    "LIVE_JPEG_QUALITY",
    85
)


# ============================================================================
# GALLERY
# ============================================================================

def refresh_gallery(force=False):

    now = time.time()

    if (
        not force
        and _gallery["matrix"] is not None
        and now - _gallery["loaded_at"]
        < GALLERY_REFRESH_SEC
    ):

        return

    ids, names, roles, matrix = (
        db_manager.load_stakeholder_gallery()
    )

    _gallery["ids"] = ids
    _gallery["names"] = names
    _gallery["roles"] = roles
    _gallery["matrix"] = matrix
    _gallery["loaded_at"] = now

    if not ids:

        log.warning(
            "Stakeholder gallery is EMPTY."
        )

    else:

        log.info(
            "Stakeholder gallery loaded: %d identities",
            len(ids)
        )


# ============================================================================
# VISIT COOLDOWN
# ============================================================================

def _should_log_visit(
    stakeholder_id
):

    now = time.time()

    last = _last_visit_log.get(
        stakeholder_id,
        0.0
    )

    if (
        now - last
        >= settings.VISIT_LOG_COOLDOWN_SEC
    ):

        _last_visit_log[
            stakeholder_id
        ] = now

        return True

    return False


# ============================================================================
# UNKNOWN PERSON DE-DUPLICATION
# ============================================================================

def _is_new_unknown(
    embedding
):

    now = time.time()

    # --------------------------------------------------------
    # Remove expired unknown embeddings
    # --------------------------------------------------------

    _recent_unknowns[:] = [

        (emb, timestamp)

        for emb, timestamp
        in _recent_unknowns

        if (
            now - timestamp
            < settings.UNKNOWN_LOG_COOLDOWN_SEC
        )
    ]

    # --------------------------------------------------------
    # Normalize current embedding
    # --------------------------------------------------------

    normalized = (
        face_recognizer.normalize(
            embedding
        )
    )

    # --------------------------------------------------------
    # Compare with recent unknowns
    # --------------------------------------------------------

    for emb, _timestamp in (
        _recent_unknowns
    ):

        similarity = float(
            np.dot(
                normalized,
                emb
            )
        )

        if (
            similarity
            >= settings.UNKNOWN_DUP_THRESHOLD
        ):

            return False

    # --------------------------------------------------------
    # Register this unknown
    # --------------------------------------------------------

    _recent_unknowns.append(
        (
            normalized,
            now
        )
    )

    return True


# ============================================================================
# FACE INSIDE PERSON
# ============================================================================

def _face_inside_person(
    face_box,
    person_boxes
):

    fx = (
        face_box[0]
        + face_box[2]
    ) / 2.0

    fy = (
        face_box[1]
        + face_box[3]
    ) / 2.0

    for person in person_boxes:

        x1, y1, x2, y2 = (
            person["box"]
        )

        if (
            x1 <= fx <= x2
            and
            y1 <= fy <= y2
        ):

            return True

    return False


# ============================================================================
# ATOMIC LIVE FRAME WRITER
# ============================================================================

def _write_latest_frame(
    frame
):

    if not settings.SAVE_LATEST_FRAME:

        return False

    final_path = (
        settings.LATEST_FRAME_PATH
    )

    # --------------------------------------------------------
    # Temporary file
    #
    # Never write directly to latest.jpg.
    # --------------------------------------------------------

    temp_path = (
        final_path
        + ".tmp.jpg"
    )

    try:

        # ----------------------------------------------------
        # Ensure directory exists
        # ----------------------------------------------------

        directory = os.path.dirname(
            final_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        # ----------------------------------------------------
        # Encode/write JPEG to temporary file
        # ----------------------------------------------------

        success = cv2.imwrite(

            temp_path,

            frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                int(LIVE_JPEG_QUALITY)
            ]

        )

        if not success:

            log.warning(
                "cv2.imwrite() failed for live frame."
            )

            return False

        # ----------------------------------------------------
        # Atomic replacement
        #
        # Streamlit will see either the old complete JPEG
        # or the new complete JPEG.
        #
        # It will never see a partially-written JPEG.
        # ----------------------------------------------------

        os.replace(
            temp_path,
            final_path
        )

        return True

    except Exception as exc:

        log.warning(
            "Could not write latest frame: %s",
            exc
        )

        # ----------------------------------------------------
        # Clean temporary file
        # ----------------------------------------------------

        try:

            if os.path.exists(
                temp_path
            ):

                os.remove(
                    temp_path
                )

        except Exception:

            pass

        return False


# ============================================================================
# DRAW PERSON DETECTIONS
# ============================================================================

def _draw_person_detections(
    frame,
    persons
):

    for det in persons:

        box = det.get(
            "box"
        )

        confidence = float(
            det.get(
                "conf",
                0.0
            )
        )

        if box is None:

            continue

        image_utils.draw_box(

            frame,

            box,

            f"person {confidence:.2f}",

            image_utils.COLOR_PERSON

        )


# ============================================================================
# DRAW FACE DETECTIONS
# ============================================================================

def _draw_face_detections(
    frame,
    faces
):

    for face in faces:

        box = face.get(
            "box"
        )

        if box is None:

            continue

        label = face.get(
            "label",
            "UNKNOWN"
        )

        color = face.get(
            "color",
            image_utils.COLOR_UNKNOWN
        )

        image_utils.draw_box(

            frame,

            box,

            label,

            color

        )


# ============================================================================
# PROCESS FRAME
# ============================================================================

def process_frame(
    frame,
    camera_location
):

    """
    Run YOLO + InsightFace on one frame.

    Returns:
        summary
        persons
        faces_for_display

    The original frame is NOT returned here.

    This allows the main loop to maintain a clean camera frame
    and draw the latest known detections on every frame.
    """

    refresh_gallery()

    # ========================================================
    # YOLO PERSON DETECTION
    # ========================================================

    persons = (
        person_detector.detect_persons(
            frame
        )
    )

    # ========================================================
    # INSIGHTFACE
    # ========================================================

    detected_faces = (
        face_recognizer.extract_faces(
            frame
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "persons": len(persons),

        "recognized": 0,

        "unknown": 0,
    }

    # ========================================================
    # FACE DISPLAY RESULTS
    # ========================================================

    faces_for_display = []

    # ========================================================
    # PROCESS FACES
    # ========================================================

    for face in detected_faces:

        # ----------------------------------------------------
        # Check whether face belongs to detected person
        # ----------------------------------------------------

        if (
            persons
            and not _face_inside_person(
                face["box"],
                persons
            )
        ):

            continue

        # ----------------------------------------------------
        # Match face against gallery
        # ----------------------------------------------------

        idx, similarity, is_match = (
            face_recognizer.match_embedding(

                face["embedding"],

                _gallery["matrix"]

            )
        )

        # ====================================================
        # KNOWN PERSON
        # ====================================================

        if is_match:

            summary[
                "recognized"
            ] += 1

            stakeholder_id = (
                _gallery["ids"][idx]
            )

            name = (
                _gallery["names"][idx]
            )

            role = (
                _gallery["roles"][idx]
            )

            label = (
                f"{name} ({role}) "
                f"{similarity:.2f}"
            )

            faces_for_display.append({

                "box": face["box"],

                "label": label,

                "color":
                    image_utils.COLOR_STAKEHOLDER,

            })

            # ------------------------------------------------
            # Visit logging
            # ------------------------------------------------

            if _should_log_visit(
                stakeholder_id
            ):

                db_manager.log_visit(

                    stakeholder_id,

                    camera_location,

                    similarity

                )

                log.info(

                    "VISIT | %s (%s) @ %s sim=%.3f",

                    name,

                    role,

                    camera_location,

                    similarity

                )

        # ====================================================
        # UNKNOWN PERSON
        # ====================================================

        else:

            summary[
                "unknown"
            ] += 1

            label = (
                f"UNKNOWN "
                f"{similarity:.2f}"
            )

            faces_for_display.append({

                "box": face["box"],

                "label": label,

                "color":
                    image_utils.COLOR_UNKNOWN,

            })

            # ------------------------------------------------
            # Save genuinely new unknown
            # ------------------------------------------------

            if _is_new_unknown(
                face["embedding"]
            ):

                path = (
                    image_utils.save_face_crop(

                        frame,

                        face["box"],

                        settings.UNKNOWN_IMG_DIR,

                        "unknown"

                    )
                )

                if path is not None:

                    db_manager.log_unknown(

                        path,

                        face["embedding"],

                        camera_location

                    )

                    log.warning(

                        "UNKNOWN person registered "
                        "@ %s -> %s",

                        camera_location,

                        path

                    )

    return (
        summary,
        persons,
        faces_for_display
    )


# ============================================================================
# SURVEILLANCE
# ============================================================================

def run_surveillance(
    source=None,
    camera_location=None,
    display=None,
    max_frames=None
):

    # ========================================================
    # SETUP
    # ========================================================

    settings.ensure_directories()

    db_manager.init_db()

    source = (
        settings.DEFAULT_SOURCE
        if source is None
        else source
    )

    camera_location = (

        camera_location

        or
        settings.DEFAULT_CAMERA_LOCATION

    )

    display = (

        settings.DISPLAY_WINDOW

        if display is None

        else display

    )

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    log.info(
        "Initializing YOLO..."
    )

    person_detector.init_detector()

    log.info(
        "Initializing InsightFace..."
    )

    face_recognizer.init_face_model()

    refresh_gallery(
        force=True
    )

    # ========================================================
    # CAMERA
    # ========================================================

    try:

        stream = (
            webcam_stream.VideoStream(
                source
            )
        )

    except Exception as exc:

        log.error(
            "Could not open stream: %s",
            exc
        )

        return

    log.info(
        "================================================"
    )

    log.info(
        "Surveillance started"
    )

    log.info(
        "Source: %r",
        source
    )

    log.info(
        "Location: %s",
        camera_location
    )

    log.info(
        "AI device: %s",
        settings.AI_DEVICE
    )

    log.info(
        "Frame processing interval: every %d frame(s)",
        settings.FRAME_PROCESS_EVERY_N
    )

    log.info(
        "Live frame output: %.1f FPS",
        LIVE_FRAME_FPS
    )

    log.info(
        "================================================"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    frame_idx = 0

    processed_frames = 0

    failures = 0

    # ========================================================
    # FPS TRACKING
    # ========================================================

    fps_start_time = time.time()

    fps_frame_count = 0

    camera_fps = 0.0

    # ========================================================
    # LIVE JPEG TIMING
    # ========================================================

    last_live_write = 0.0

    live_interval = (
        1.0
        / max(
            LIVE_FRAME_FPS,
            1.0
        )
    )

    # ========================================================
    # LAST AI RESULTS
    #
    # These are intentionally kept between AI inference
    # frames. This prevents the bounding boxes from
    # disappearing every other frame.
    # ========================================================

    last_persons = []

    last_faces = []

    last_summary = {

        "persons": 0,

        "recognized": 0,

        "unknown": 0,
    }

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # =================================================
            # GET LATEST CAMERA FRAME
            # =================================================

            ok, frame = (
                stream.read()
            )

            if not ok:

                failures += 1

                if failures >= 30:

                    log.error(
                        "Camera stream unavailable."
                    )

                    break

                time.sleep(
                    0.005
                )

                continue

            failures = 0

            frame_idx += 1

            # =================================================
            # CAMERA FPS
            # =================================================

            fps_frame_count += 1

            fps_elapsed = (
                time.time()
                - fps_start_time
            )

            if fps_elapsed >= 1.0:

                camera_fps = (
                    fps_frame_count
                    / fps_elapsed
                )

                fps_frame_count = 0

                fps_start_time = (
                    time.time()
                )

            # =================================================
            # AI PROCESSING
            # =================================================

            should_process = (

                frame_idx
                % settings.FRAME_PROCESS_EVERY_N
                == 0

            )

            if should_process:

                try:

                    (
                        last_summary,
                        last_persons,
                        last_faces
                    ) = process_frame(

                        frame,

                        camera_location

                    )

                    processed_frames += 1

                except Exception as exc:

                    log.exception(
                        "AI frame processing failed: %s",
                        exc
                    )

            # =================================================
            # DRAW THE LATEST AI RESULTS
            #
            # IMPORTANT:
            # These results are drawn onto EVERY camera frame.
            #
            # Therefore:
            #
            # frame 1 → old detections
            # frame 2 → new detections
            # frame 3 → same new detections
            # frame 4 → new detections
            #
            # Instead of:
            #
            # frame 1 → boxes
            # frame 2 → no boxes
            # frame 3 → boxes
            # =================================================

            _draw_person_detections(

                frame,

                last_persons

            )

            _draw_face_detections(

                frame,

                last_faces

            )

            # =================================================
            # HEADER
            # =================================================

            image_utils.draw_header(

                frame,

                (
                    f"{camera_location} | "
                    f"Camera: {camera_fps:.1f} FPS | "
                    f"Persons: "
                    f"{last_summary['persons']} | "
                    f"Known: "
                    f"{last_summary['recognized']} | "
                    f"Unknown: "
                    f"{last_summary['unknown']} | "
                    f"AI: {settings.AI_DEVICE} | "
                    f"q = quit"
                )

            )

            # =================================================
            # WRITE LIVE FRAME
            #
            # Only encode JPEG at the dashboard FPS.
            # =================================================

            now = time.time()

            if (
                settings.SAVE_LATEST_FRAME
                and
                (
                    now
                    - last_live_write
                )
                >= live_interval
            ):

                if _write_latest_frame(
                    frame
                ):

                    last_live_write = now

            # =================================================
            # OPTIONAL LOCAL DISPLAY
            # =================================================

            if display:

                cv2.imshow(

                    "Campus Surveillance",

                    frame

                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    log.info(
                        "Quit requested."
                    )

                    break

            # =================================================
            # MAX FRAMES
            # =================================================

            if (
                max_frames is not None
                and processed_frames
                >= max_frames
            ):

                break

    except KeyboardInterrupt:

        log.info(
            "Surveillance interrupted."
        )

    except Exception as exc:

        log.exception(
            "Surveillance loop failed: %s",
            exc
        )

    finally:

        # ====================================================
        # RELEASE CAMERA
        # ====================================================

        try:

            stream.release()

        except Exception as exc:

            log.warning(
                "Could not release camera: %s",
                exc
            )

        # ====================================================
        # CLOSE OPENCV
        # ====================================================

        if display:

            try:

                cv2.destroyAllWindows()

            except Exception:

                pass

        log.info(
            "================================================"
        )

        log.info(
            "Surveillance stopped."
        )

        log.info(
            "Camera frames: %d",
            frame_idx
        )

        log.info(
            "AI frames: %d",
            processed_frames
        )

        log.info(
            "================================================"
        )