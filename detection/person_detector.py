"""
detection/person_detector.py

YOLO person detection optimized for Apple Silicon MPS.
"""

from ultralytics import YOLO

from config import settings
from utils.logger import get_logger


log = get_logger()


# ============================================================================
# GLOBAL MODEL
# ============================================================================

_model = None


# ============================================================================
# INITIALIZE YOLO
# ============================================================================

def init_detector(model_path=None):

    global _model

    path = (
        model_path
        or settings.YOLO_MODEL_PATH
    )

    log.info(
        "Loading YOLOv8 model: %s",
        path
    )

    _model = YOLO(path)

    log.info(
        "YOLO device: %s",
        settings.AI_DEVICE
    )

    return _model


# ============================================================================
# DETECT PERSONS
# ============================================================================

def detect_persons(
    frame,
    conf=None
):

    global _model

    # --------------------------------------------------------
    # Load model if necessary
    # --------------------------------------------------------

    if _model is None:

        init_detector()

    confidence = (
        conf
        if conf is not None
        else settings.YOLO_CONF_THRESHOLD
    )

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = _model.predict(

        frame,

        # Apple MPS
        device=settings.AI_DEVICE,

        # 640x640
        imgsz=settings.YOLO_IMG_SIZE,

        # Confidence
        conf=confidence,

        # Only person class
        classes=[
            settings.YOLO_PERSON_CLASS
        ],

        # Don't print per-frame output
        verbose=False,

        # Don't save anything
        save=False,

        # Don't show anything
        show=False,
    )

    # --------------------------------------------------------
    # Convert results
    # --------------------------------------------------------

    detections = []

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            confidence_value = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            detections.append({

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "conf": confidence_value
            })

    return detections