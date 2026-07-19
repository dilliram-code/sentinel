"""
detection/person_detector.py
----------------------------
YOLOv8 person detection (Jocher et al., 2023) in modular/functional style.
The model handle is kept as module-level state and lazy-loaded once.
"""

from ultralytics import YOLO

from config import settings
from utils.logger import get_logger

log = get_logger()

_model = None  # module-level model handle (loaded once)


def init_detector(model_path=None):
    """
    Load the YOLOv8 model. Called automatically by detect_persons() if needed.
    Pass a fine-tuned checkpoint path to use a campus-specific model.
    """
    global _model
    path = model_path or settings.YOLO_MODEL_PATH
    log.info("Loading YOLOv8 model: %s", path)
    _model = YOLO(path)
    return _model


def detect_persons(frame, conf=None):
    """
    Detect persons in a BGR frame.

    Returns a list of dicts: {"box": (x1, y1, x2, y2), "conf": float}
    Only COCO class 0 ("person") is kept.
    """
    global _model
    if _model is None:
        init_detector()

    results = _model.predict(
        frame,
        conf=conf if conf is not None else settings.YOLO_CONF_THRESHOLD,
        classes=[settings.YOLO_PERSON_CLASS],
        imgsz=settings.YOLO_IMG_SIZE,
        verbose=False,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "box": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
            })
    return detections
