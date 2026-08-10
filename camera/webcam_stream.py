"""
camera/webcam_stream.py
"""

import time
import threading

import cv2

from config import settings
from utils.logger import get_logger


log = get_logger()


# ============================================================================
# SOURCE
# ============================================================================

def resolve_source(source):

    if isinstance(source, int):

        return source

    text = str(
        source
    ).strip()

    if text.isdigit():

        return int(text)

    return text


# ============================================================================
# OPEN WEBCAM
# ============================================================================

def _open_index(index):

    log.info(
        "Trying webcam index %d...",
        index
    )

    # --------------------------------------------------------
    # macOS AVFoundation
    # --------------------------------------------------------

    if hasattr(
        cv2,
        "CAP_AVFOUNDATION"
    ):

        cap = cv2.VideoCapture(
            index,
            cv2.CAP_AVFOUNDATION
        )

    else:

        cap = cv2.VideoCapture(
            index
        )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not cap.isOpened():

        cap.release()

        return None

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        settings.CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        settings.CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        settings.CAMERA_FPS
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    deadline = (
        time.time()
        + 1.0
    )

    valid_frames = 0

    while (
        time.time() < deadline
        and valid_frames < 2
    ):

        ok, frame = cap.read()

        if (
            ok
            and frame is not None
            and frame.size > 0
        ):

            valid_frames += 1

    if valid_frames == 0:

        cap.release()

        return None

    log.info(
        "Webcam %d opened: %.0fx%.0f",
        index,
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        ),
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    return cap


# ============================================================================
# FIND WEBCAM
# ============================================================================

def find_working_webcam():

    for index in range(3):

        cap = _open_index(index)

        if cap is not None:

            log.info(
                "Webcam found at index %d",
                index
            )

            return index, cap

    return None, None


# ============================================================================
# OPEN STREAM
# ============================================================================

def open_stream(source=0):

    src = resolve_source(
        source
    )

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    if isinstance(src, int):

        cap = _open_index(
            src
        )

        if cap is not None:

            return cap

        _, cap = (
            find_working_webcam()
        )

        return cap

    # --------------------------------------------------------
    # RTSP / HTTP
    # --------------------------------------------------------

    if src.lower().startswith(
        (
            "rtsp://",
            "http://",
            "https://"
        )
    ):

        log.info(
            "Opening network stream..."
        )

        cap = cv2.VideoCapture(
            src,
            cv2.CAP_FFMPEG
        )

        try:

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        except Exception:

            pass

        if not cap.isOpened():

            cap.release()

            return None

        return cap

    # --------------------------------------------------------
    # Video file
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        src
    )

    if not cap.isOpened():

        cap.release()

        return None

    return cap


# ============================================================================
# NORMAL READ
# ============================================================================

def read_frame(cap):

    if cap is None:

        return False, None

    ok, frame = cap.read()

    if not ok:

        return False, None

    if frame is None:

        return False, None

    if frame.size == 0:

        return False, None

    return True, frame


# ============================================================================
# BACKGROUND VIDEO STREAM
# ============================================================================

class VideoStream:

    def __init__(self, source=0):

        self.cap = open_stream(
            source
        )

        if self.cap is None:

            raise RuntimeError(
                f"Could not open video source: {source}"
            )

        self.frame = None

        self.lock = threading.Lock()

        self.running = True

        self.thread = threading.Thread(

            target=self._capture_loop,

            daemon=True

        )

        self.thread.start()

    # ------------------------------------------------------------------------
    # CAPTURE LOOP
    # ------------------------------------------------------------------------

    def _capture_loop(self):

        while self.running:

            try:

                ok, frame = (
                    self.cap.read()
                )

                if not ok:

                    time.sleep(
                        0.005
                    )

                    continue

                if (
                    frame is None
                    or frame.size == 0
                ):

                    continue

                # ------------------------------------------------------------
                # VERY IMPORTANT:
                #
                # Replace the old frame.
                #
                # Don't create a queue of old frames.
                # ------------------------------------------------------------

                with self.lock:

                    self.frame = frame

            except Exception as exc:

                log.exception(
                    "Camera capture error: %s",
                    exc
                )

                time.sleep(
                    0.01
                )

    # ------------------------------------------------------------------------
    # GET LATEST FRAME
    # ------------------------------------------------------------------------

    def read(self):

        with self.lock:

            if self.frame is None:

                return False, None

            frame = self.frame.copy()

        return True, frame

    # ------------------------------------------------------------------------
    # RELEASE
    # ------------------------------------------------------------------------

    def release(self):

        self.running = False

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=1.0
            )

        if self.cap is not None:

            self.cap.release()

        self.cap = None

        self.frame = None


# ============================================================================
# COMPATIBILITY FUNCTION
# ============================================================================

def release_stream(cap):

    """
    Compatibility function.

    The updated surveillance pipeline uses VideoStream.release().
    This is kept so other parts of the project don't immediately break.
    """

    if cap is None:

        return

    try:

        cap.release()

    except Exception:

        pass