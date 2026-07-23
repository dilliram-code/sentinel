"""
config/settings.py
------------------
Central configuration for the Campus Surveillance System.
All modules import their constants from here so tuning happens in ONE place.
"""

import os

# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

# enrolled face images
STAKEHOLDER_IMG_DIR = os.path.join(DATA_DIR, "stakeholders")  

# auto-captured unknowns  
UNKNOWN_IMG_DIR = os.path.join(DATA_DIR, "unknown_faces")    

# latest annotated frame (for dashboard)
LIVE_FRAME_DIR      = os.path.join(DATA_DIR, "live")             
MODELS_DIR          = os.path.join(BASE_DIR, "models")
DB_PATH             = os.path.join(DATA_DIR, "campus_surveillance.db")
LOG_FILE            = os.path.join(DATA_DIR, "system.log")
LATEST_FRAME_PATH   = os.path.join(LIVE_FRAME_DIR, "latest.jpg")


# YOLOv8 person detection
YOLO_MODEL_PATH     = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.45          # min confidence for a person box
YOLO_PERSON_CLASS   = 0             # COCO class id for "person"
YOLO_IMG_SIZE       = 640

# InsightFace recognition
# ---------------------------------------------------------------------------
FACE_MODEL_NAME     = "buffalo_l"   # InsightFace model pack (auto-downloaded)
FACE_DET_SIZE       = (640, 640)
FACE_PROVIDERS      = ["CPUExecutionProvider"]  # switch to ["CUDAExecutionProvider"] on GPU


FACE_MATCH_THRESHOLD = 0.45

# Logging / de-duplication behaviour
# ---------------------------------------------------------------------------
VISIT_LOG_COOLDOWN_SEC   = 60   # don't re-log the same stakeholder within N seconds
UNKNOWN_DUP_THRESHOLD    = 0.50 # cosine sim above which an unknown is "the same" unknown
UNKNOWN_LOG_COOLDOWN_SEC = 120  # don't re-save the same unknown within N seconds

# DEFAULT_SOURCE may be:
#   0                              -> built-in laptop webcam (DEFAULT)
#   1, 2, ...                      -> external USB webcams
#   "path/to/video.mp4"            -> recorded footage (testing without camera)
#   "rtsp://user:pass@ip:554/..."  -> future upgrade to a real IP/CCTV camera
# If the chosen index fails, camera/webcam_stream.py auto-probes 0..2.
DEFAULT_SOURCE          = 0
DEFAULT_CAMERA_LOCATION = "Laptop Webcam"
FRAME_PROCESS_EVERY_N   = 2     # process every Nth frame (real-time speed-up)
DISPLAY_WINDOW          = True  # cv2.imshow preview (set False on headless servers)
SAVE_LATEST_FRAME       = True  # write annotated frame for the Streamlit dashboard


# Helpers
# ---------------------------------------------------------------------------
def ensure_directories():
    """Create every data directory the system needs. Safe to call repeatedly."""
    for path in (DATA_DIR, STAKEHOLDER_IMG_DIR, UNKNOWN_IMG_DIR,
                LIVE_FRAME_DIR, MODELS_DIR):
        os.makedirs(path, exist_ok=True)
