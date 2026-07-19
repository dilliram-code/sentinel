"""
main.py
-------
Command-line entry point for the Campus Surveillance System.

Webcam workflow (in order):
    python main.py init-db                      # 1. create DB + folders
    python main.py check                        # 2. verify libs/models/webcam
    python main.py register --uid S001 --name "Sita Sharma" \
                            --role Student --webcam         # 3. enroll people
    python main.py list                         # 4. confirm registration
    python main.py run                          # 5. start surveillance (q = quit)
    streamlit run dashboard/app.py              # 6. dashboard (second terminal)

Registration from photo files also works:
    python main.py register --uid F010 --name "Dr. Ram K." --role Faculty \
                            --images ./photos/ram/
"""

import argparse

from config import settings
from database import db_manager
from utils.logger import get_logger

log = get_logger()


def cmd_init_db(_args):
    settings.ensure_directories()
    db_manager.init_db()
    log.info("Database initialized at %s", settings.DB_PATH)


def cmd_register(args):
    # Import here so `init-db` / `list` work without ML deps installed.
    from registration import register_stakeholder

    if args.webcam:
        register_stakeholder.register_from_webcam(
            args.uid, args.name, args.role,
            camera_index=args.camera_index, samples=args.samples)
    elif args.images:
        register_stakeholder.register_from_images(
            args.uid, args.name, args.role, args.images)
    else:
        log.error("Provide --images PATH or --webcam for registration.")


def cmd_run(args):
    from pipeline import surveillance_pipeline

    surveillance_pipeline.run_surveillance(
        source=args.source,
        camera_location=args.location,
        display=not args.no_display,
        max_frames=args.max_frames,
    )


def cmd_check(_args):
    """
    Pre-flight self-check: verifies libraries, models, database and webcam.
    """
    ok = True
    print("=" * 60)
    print("CAMPUS SURVEILLANCE — ENVIRONMENT CHECK")
    print("=" * 60)

    # 1. Libraries -------------------------------------------------------
    for lib in ("cv2", "numpy", "pandas", "ultralytics", "insightface",
                "onnxruntime", "streamlit"):
        try:
            __import__(lib)
            print(f"[OK]   library '{lib}' importable")
        except ImportError as exc:
            ok = False
            print(f"[FAIL] library '{lib}' missing -> pip install -r "
                  f"requirements.txt   ({exc})")

    # 2. Folders + database ---------------------------------------------
    try:
        settings.ensure_directories()
        db_manager.init_db()
        print(f"[OK]   folders + database ready ({settings.DB_PATH})")
    except Exception as exc:
        ok = False
        print(f"[FAIL] database init: {exc}")

    # 3. YOLOv8 (downloads yolov8n.pt on first run — needs internet once)
    try:
        from detection import person_detector
        person_detector.init_detector()
        print("[OK]   YOLOv8 model loaded")
    except Exception as exc:
        ok = False
        print(f"[FAIL] YOLOv8: {exc}")

    # 4. InsightFace (downloads buffalo_l pack on first run) --------------
    try:
        from recognition import face_recognizer
        face_recognizer.init_face_model()
        print("[OK]   InsightFace model loaded")
    except Exception as exc:
        ok = False
        print(f"[FAIL] InsightFace: {exc}")

    # 5. Webcam ----------------------------------------------------------
    try:
        from camera import webcam_stream
        idx, cap = webcam_stream.find_working_webcam()
        if cap is not None:
            webcam_stream.release_stream(cap)
            print(f"[OK]   webcam working at index {idx}")
        else:
            ok = False
            print("[FAIL] no webcam found — close apps using the camera "
                  "(Zoom/Teams/browser) and check OS camera permissions")
    except Exception as exc:
        ok = False
        print(f"[FAIL] webcam test: {exc}")

    print("=" * 60)
    print("ALL CHECKS PASSED ✅ — you are ready to register and run."
          if ok else
          "SOME CHECKS FAILED ❌ — fix the [FAIL] lines above first.")
    print("=" * 60)


def cmd_list(_args):
    db_manager.init_db()
    rows = db_manager.list_stakeholders()
    if not rows:
        print("No stakeholders registered yet.")
        return
    print(f"{'ID':<4} {'UID':<10} {'Name':<25} {'Role':<10} Registered")
    for rid, uid, name, role, _img, reg in rows:
        print(f"{rid:<4} {uid:<10} {name:<25} {role:<10} {reg}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-Time Campus Stakeholder Identification "
                    "(YOLOv8 + InsightFace)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create the SQLite database & folders")
    sub.add_parser("check", help="Verify libraries, models, DB and webcam")

    reg = sub.add_parser("register", help="Enroll a stakeholder")
    reg.add_argument("--uid", required=True, help="Unique ID, e.g. S001")
    reg.add_argument("--name", required=True)
    reg.add_argument("--role", required=True,
                     choices=["Student", "Faculty", "Staff", "Authorized"])
    reg.add_argument("--images", help="Image file / folder / glob pattern")
    reg.add_argument("--webcam", action="store_true",
                     help="Capture enrollment samples from a webcam")
    reg.add_argument("--camera-index", type=int, default=0)
    reg.add_argument("--samples", type=int, default=5)

    run = sub.add_parser("run", help="Start real-time surveillance")
    run.add_argument("--source", default=None,
                     help="Webcam index, RTSP/HTTP URL, or video file "
                          "(default: config DEFAULT_SOURCE)")
    run.add_argument("--location", default=None,
                     help="Camera location label (default: config)")
    run.add_argument("--no-display", action="store_true",
                     help="Headless mode (no cv2 preview window)")
    run.add_argument("--max-frames", type=int, default=None,
                     help="Stop after N processed frames (evaluation runs)")

    sub.add_parser("list", help="List registered stakeholders")
    return parser


def main():
    args = build_parser().parse_args()
    {"init-db": cmd_init_db,
     "check": cmd_check,
     "register": cmd_register,
     "run": cmd_run,
     "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    main()
