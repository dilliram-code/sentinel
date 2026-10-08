"""
main.py
-------
Command-line entry point for the Campus Surveillance System.

Workflow:
    1. Initialize DB:     python main.py init-db
    2. Self Check:        python main.py check
    3. Web Dashboard:     python main.py web        (Access at http://localhost:8000)
    4. Register person:   python main.py register --uid S001 --name "Sita Sharma" --role Student --webcam
    5. List stakeholders: python main.py list
    6. Terminal Runner:   python main.py run
"""

import argparse
import sys

from config import settings
from database import db_manager
from utils.logger import get_logger

log = get_logger()


def cmd_init_db(_args):
    settings.ensure_directories()
    db_manager.init_db()
    log.info("Database initialized successfully at %s", settings.DB_PATH)


def cmd_register(args):
    from registration import register_stakeholder

    if args.webcam:
        register_stakeholder.register_from_webcam(
            args.uid, args.name, args.role,
            camera_index=args.camera_index, samples=args.samples
        )
    elif args.images:
        register_stakeholder.register_from_images(
            args.uid, args.name, args.role, args.images
        )
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


def cmd_web(args):
    """Launch the modern high-performance web dashboard."""
    import uvicorn
    import server

    print(f"\n=======================================================")
    print(f"🚀 CAMPUS SENTINEL — WEB DASHBOARD RUNNING")
    print(f"👉 Open in browser: http://localhost:{args.port}")
    print(f"=======================================================\n")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


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
                "onnxruntime", "fastapi", "uvicorn", "websockets"):
        try:
            __import__(lib)
            print(f"[OK]   library '{lib}' importable")
        except ImportError as exc:
            ok = False
            print(f"[FAIL] library '{lib}' missing -> pip install -r requirements.txt ({exc})")

    # 2. Folders + database ---------------------------------------------
    try:
        settings.ensure_directories()
        db_manager.init_db()
        print(f"[OK]   folders + database ready ({settings.DB_PATH})")
    except Exception as exc:
        ok = False
        print(f"[FAIL] database init: {exc}")

    # 3. YOLOv8
    try:
        from detection import person_detector
        person_detector.init_detector()
        print("[OK]   YOLOv8 model loaded")
    except Exception as exc:
        ok = False
        print(f"[FAIL] YOLOv8: {exc}")

    # 4. InsightFace
    try:
        from recognition import face_recognizer
        face_recognizer.init_face_model()
        print("[OK]   InsightFace model loaded")
    except Exception as exc:
        ok = False
        print(f"[FAIL] InsightFace: {exc}")

    # 5. Webcam
    try:
        from camera import webcam_stream
        idx, cap = webcam_stream.find_working_webcam()
        if cap is not None:
            webcam_stream.release_stream(cap)
            print(f"[OK]   webcam working at index {idx}")
        else:
            print("[WARN] no local webcam opened (fine if using RTSP/network feed)")
    except Exception as exc:
        print(f"[WARN] webcam check: {exc}")

    print("=" * 60)
    print("ALL CHECKS PASSED ✅ — ready to launch web dashboard." if ok else "SOME CHECKS FAILED ❌ — please review above.")
    print("=" * 60)


def cmd_list(_args):
    db_manager.init_db()
    rows = db_manager.list_stakeholders()
    if not rows:
        print("No stakeholders registered yet.")
        return
    print(f"{'ID':<4} {'UID':<10} {'Name':<25} {'Role':<12} Registered")
    print("-" * 65)
    for rid, uid, name, role, _img, reg in rows:
        print(f"{rid:<4} {uid:<10} {name:<25} {role:<12} {reg}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Campus Sentinel — Real-Time AI Stakeholder Identification System"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init-db
    sub.add_parser("init-db", help="Create the SQLite database & folders")
    
    # check
    sub.add_parser("check", help="Verify libraries, models, DB and camera")

    # web / dashboard
    web_p = sub.add_parser("web", help="Start the modern web dashboard (FastAPI + WebSockets)")
    web_p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    web_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    web_p.add_argument("--reload", action="store_true", help="Auto-reload for development")

    # register
    reg = sub.add_parser("register", help="Enroll a stakeholder")
    reg.add_argument("--uid", required=True, help="Unique ID, e.g. S001")
    reg.add_argument("--name", required=True)
    reg.add_argument("--role", required=True, choices=["Student", "Faculty", "Staff", "Authorized"])
    reg.add_argument("--images", help="Image file / folder / glob pattern")
    reg.add_argument("--webcam", action="store_true", help="Capture enrollment samples from a webcam")
    reg.add_argument("--camera-index", type=int, default=0)
    reg.add_argument("--samples", type=int, default=5)

    # run (headless or terminal runner)
    run = sub.add_parser("run", help="Start real-time surveillance in terminal/preview mode")
    run.add_argument("--source", default=None, help="Webcam index, RTSP/HTTP URL, or video file")
    run.add_argument("--location", default=None, help="Camera location label")
    run.add_argument("--no-display", action="store_true", help="Headless mode (no cv2 preview window)")
    run.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")

    # list
    sub.add_parser("list", help="List registered stakeholders")
    return parser


def main():
    args = build_parser().parse_args()
    dispatch = {
        "init-db": cmd_init_db,
        "check": cmd_check,
        "web": cmd_web,
        "register": cmd_register,
        "run": cmd_run,
        "list": cmd_list
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
