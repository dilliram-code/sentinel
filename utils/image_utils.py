"""
utils/image_utils.py

"""

import os
from datetime import datetime

import cv2

# BGR colours
COLOR_STAKEHOLDER = (0, 200, 0)     # green
COLOR_UNKNOWN     = (0, 0, 220)     # red
COLOR_PERSON      = (200, 160, 0)   # cyan-ish for raw person boxes


def clamp_box(box, width, height):
    """Clamp a (x1, y1, x2, y2) box to image bounds and return ints."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def crop_region(frame, box, margin=0.15):
    """
    Crop a box from the frame with a relative margin (context around the face).
    Returns the crop, or None if the box is degenerate.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1, x2, y2 = clamp_box((x1 - mx, y1 - my, x2 + mx, y2 + my), w, h)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def save_face_crop(frame, box, out_dir, prefix="unknown"):
    """
    Save the face crop of `box` into `out_dir` with a timestamped filename.
    Returns the saved path, or None on failure.
    """
    crop = crop_region(frame, box)
    if crop is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(out_dir, f"{prefix}_{stamp}.jpg")
    return path if cv2.imwrite(path, crop) else None


def draw_box(frame, box, label, color):
    """Draw one labelled bounding box in place."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 8, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
    cv2.putText(frame, label, (x1 + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_header(frame, text):
    """Draw a translucent status bar with `text` at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 32), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
