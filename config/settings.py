"""
config/settings.py

Central configuration for the Campus Surveillance System.
Optimized for Apple Silicon / MacBook M2.
"""

import os
import torch


# ============================================================================
# BASE PATHS
# ============================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

STAKEHOLDER_IMG_DIR = os.path.join(
    DATA_DIR,
    "stakeholders"
)

UNKNOWN_IMG_DIR = os.path.join(
    DATA_DIR,
    "unknown_faces"
)

LIVE_FRAME_DIR = os.path.join(
    DATA_DIR,
    "live"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DB_PATH = os.path.join(
    DATA_DIR,
    "campus_surveillance.db"
)

LOG_FILE = os.path.join(
    DATA_DIR,
    "system.log"
)

LATEST_FRAME_PATH = os.path.join(
    LIVE_FRAME_DIR,
    "latest.jpg"
)


# ============================================================================
# HARDWARE / DEVICE
# ============================================================================

# ------------------------------------------------------------
# Detect Apple MPS
# ------------------------------------------------------------

if torch.backends.mps.is_available():

    AI_DEVICE = "mps"

elif torch.cuda.is_available():

    AI_DEVICE = "cuda"

else:

    AI_DEVICE = "cpu"


print(
    f"[SYSTEM] AI device: {AI_DEVICE}"
)


# ============================================================================
# YOLO
# ============================================================================

YOLO_MODEL_PATH = "yolov8n.pt"

YOLO_CONF_THRESHOLD = 0.45

YOLO_PERSON_CLASS = 0

LIVE_FRAME_FPS = 12
LIVE_JPEG_QUALITY = 85
# Smaller image = faster inference
YOLO_IMG_SIZE = 640

# Process YOLO every Nth frame
#
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
#
FRAME_PROCESS_EVERY_N = 2


# ============================================================================
# INSIGHTFACE
# ============================================================================

# buffalo_l = better accuracy but slower
#
# buffalo_s = considerably faster and recommended for real-time
#
FACE_MODEL_NAME = "buffalo_s"

FACE_DET_SIZE = (
    640,
    640
)


# ------------------------------------------------------------
# InsightFace / ONNX Runtime providers
#
# CoreML is the Apple Silicon acceleration backend.
# CPU is kept as fallback.
# ------------------------------------------------------------

try:

    import onnxruntime as ort

    AVAILABLE_ORT_PROVIDERS = (
        ort.get_available_providers()
    )

except Exception:

    AVAILABLE_ORT_PROVIDERS = []


if "CoreMLExecutionProvider" in AVAILABLE_ORT_PROVIDERS:

    FACE_PROVIDERS = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

else:

    FACE_PROVIDERS = [
        "CPUExecutionProvider"
    ]


print(
    f"[SYSTEM] InsightFace providers: "
    f"{FACE_PROVIDERS}"
)


# ============================================================================
# FACE MATCHING
# ============================================================================

FACE_MATCH_THRESHOLD = 0.45


# ============================================================================
# UNKNOWN PERSON DE-DUPLICATION
# ============================================================================

UNKNOWN_DUP_THRESHOLD = 0.50

UNKNOWN_LOG_COOLDOWN_SEC = 120


# ============================================================================
# VISIT LOGGING
# ============================================================================

VISIT_LOG_COOLDOWN_SEC = 60


# ============================================================================
# CAMERA
# ============================================================================

DEFAULT_SOURCE = 0

DEFAULT_CAMERA_LOCATION = "Laptop Webcam"


# Camera resolution
CAMERA_WIDTH = 1280

CAMERA_HEIGHT = 720

CAMERA_FPS = 30


# ============================================================================
# DISPLAY
# ============================================================================

DISPLAY_WINDOW = True

SAVE_LATEST_FRAME = True


# ============================================================================
# PERFORMANCE
# ============================================================================

# Keep camera capture independent from AI processing
USE_BACKGROUND_CAPTURE = True

# Only keep the newest camera frame
LATEST_FRAME_ONLY = True


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories():

    directories = (
        DATA_DIR,
        STAKEHOLDER_IMG_DIR,
        UNKNOWN_IMG_DIR,
        LIVE_FRAME_DIR,
        MODELS_DIR,
    )

    for path in directories:

        os.makedirs(
            path,
            exist_ok=True
        )