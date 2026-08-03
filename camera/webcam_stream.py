"""
camera/webcam_stream.py
-----------------------
for CCTV camera: pass RTSP URL as the source
"""

import time

import cv2

from utils.logger import get_logger

log = get_logger()

# How many webcam indices to probe when auto-detecting (0 = built-in laptop
# cam on almost every machine; 1, 2 = external USB cams).
MAX_PROBE_INDEX = 3
WARMUP_FRAMES = 2          # discard first frames (auto-exposure settling)
WARMUP_TIMEOUT_SEC = 1.0


def resolve_source(source):
    """
    Normalize a user-supplied source:
    """
    if isinstance(source, int):
        return source
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def _open_index(index):
    """Try to open one webcam index and pull a real frame. None on failure."""
    # cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) # for mac only
    cap = cv2.VideoCapture(index)                           # default
    if not cap.isOpened():
        cap.release()
        return None
    # Request a sane resolution (the driver picks the closest supported one).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm-up: some webcams return black/empty frames for the first reads.
    deadline = time.time() + WARMUP_TIMEOUT_SEC
    good = 0
    while time.time() < deadline and good < WARMUP_FRAMES:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            good += 1
    if good == 0:
        cap.release()
        return None
    return cap


def find_working_webcam():
    """Probe indices 0..MAX_PROBE_INDEX-1 and return (index, capture) or (None, None)."""
    for idx in range(MAX_PROBE_INDEX):
        cap = _open_index(idx)
        if cap is not None:
            log.info("Webcam found at index %d", idx)
            return idx, cap
        log.info("No camera at index %d, trying next...", idx)
    return None, None


def open_stream(source=0):
    """
    Open the video source and return a cv2.VideoCapture, or None on failure.
    """
    src = resolve_source(source)

    # --- webcam ---------------------------------------------------------
    if isinstance(src, int):
        cap = _open_index(src)
        if cap is not None:
            log.info("Webcam opened at index %d", src)
            return cap
        log.warning("Webcam index %d failed — auto-detecting...", src)
        idx, cap = find_working_webcam()
        if cap is not None:
            return cap
        log.error(
            "No working webcam found."
        )
        return None

    # --- network stream (rtsp/http) --------------------------------------
    if src.lower().startswith(("rtsp://", "http://", "https://")):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            log.error("Could not open network stream: %s", src)
            return None
        log.info("Network stream opened: %s", src)
        return cap

    # --- video file -------------------------------------------------------
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Could not open video file: %s (check the path)", src)
        return None
    log.info("Video file opened: %s", src)
    return cap


def read_frame(cap):
    """Read one frame. Returns (ok, frame)."""
    if cap is None:
        return False, None
    ok, frame = cap.read()
    if ok and (frame is None or frame.size == 0):
        return False, None
    return ok, frame


def release_stream(cap):
    """Release the capture handle (safe on None)."""
    if cap is not None:
        cap.release()
