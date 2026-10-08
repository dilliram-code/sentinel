"""
camera/webcam_stream.py
-----------------------
Robust camera and video stream ingestion module.
Supports macOS AVFoundation, USB webcams, RTSP/HTTP network streams, and video files.
Includes background threaded reading to eliminate buffer latency and automatic stream reconnection.
"""

import time
import threading
import cv2

from config import settings
from utils.logger import get_logger

log = get_logger()


def resolve_source(source):
    """Normalize webcam index integers, device paths, RTSP/HTTP URLs, or video file paths."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


def _open_index(index):
    """Attempt to open a local webcam at a specific device index with platform acceleration."""
    log.info("Attempting to open webcam index %d...", index)

    # Use AVFoundation on macOS if available
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        return None

    # Configure optimal camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Sensor warm-up check
    deadline = time.time() + 1.2
    valid_frames = 0
    while time.time() < deadline and valid_frames < 2:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            valid_frames += 1
        time.sleep(0.05)

    if valid_frames == 0:
        cap.release()
        return None

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.info("Webcam %d opened successfully (Resolution: %.0fx%.0f)", index, actual_w, actual_h)
    return cap


def find_working_webcam():
    """Scan local camera indices 0..3 to locate an active webcam."""
    for index in range(4):
        cap = _open_index(index)
        if cap is not None:
            return index, cap
    return None, None


def open_stream(source=0):
    """Open an appropriate VideoCapture for a given webcam index, network URL, or file."""
    src = resolve_source(source)

    # 1. Local Webcam
    if isinstance(src, int):
        cap = _open_index(src)
        if cap is not None:
            return cap
        # Fallback search if requested index is unavailable
        idx, cap = find_working_webcam()
        if cap is not None:
            log.info("Fell back to working webcam index %d", idx)
            return cap
        return None

    # 2. Network Stream (RTSP / HTTP / HTTPS)
    if src.lower().startswith(("rtsp://", "http://", "https://")):
        log.info("Opening network video stream: %s", src)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    # 3. Video File
    log.info("Opening local video file: %s", src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


class VideoStream:
    """
    Threaded, non-blocking video stream capture.
    Maintains a single newest frame buffer with thread-safe locking and handles automatic reconnection.
    """

    def __init__(self, source=0, reconnect_interval=3.0):
        self.source = source
        self.reconnect_interval = reconnect_interval
        self.cap = open_stream(self.source)
        if self.cap is None:
            raise RuntimeError(f"Could not open video stream source: {source}")

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.is_network = isinstance(self.source, str) and self.source.lower().startswith(("rtsp://", "http://", "https://"))

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _reconnect(self):
        """Attempt to re-initialize a dropped stream."""
        if not self.running:
            return False
        log.warning("Attempting to reconnect stream %r...", self.source)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        
        new_cap = open_stream(self.source)
        if new_cap is not None and new_cap.isOpened():
            self.cap = new_cap
            log.info("Stream reconnection succeeded for %r", self.source)
            return True
        return False

    def _capture_loop(self):
        """Continuous background thread pulling only the newest frames."""
        consecutive_failures = 0

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(self.reconnect_interval)
                    self._reconnect()
                    continue

                ok, frame = self.cap.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        if self.is_network:
                            log.warning("Network stream lost after 30 failed reads. Attempting reconnect.")
                            time.sleep(1.0)
                            self._reconnect()
                            consecutive_failures = 0
                        else:
                            time.sleep(0.01)
                    else:
                        time.sleep(0.005)
                    continue

                consecutive_failures = 0
                with self.lock:
                    self.frame = frame

            except Exception as exc:
                log.exception("Camera capture loop error: %s", exc)
                time.sleep(0.05)

    def read(self):
        """Fetch a copy of the latest captured frame."""
        with self.lock:
            if self.frame is None:
                return False, None
            frame = self.frame.copy()
        return True, frame

    def release(self):
        """Stop background thread and cleanly release hardware resources."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.frame = None
        log.info("VideoStream released.")


def release_stream(cap):
    """Compatibility release wrapper for standard cv2.VideoCapture instances."""
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass