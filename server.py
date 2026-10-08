"""
server.py
---------
Production-grade FastAPI application and WebSocket server for Campus Sentinel.
Provides low-latency binary video streaming, REST API endpoints, and static asset serving.
"""

import os
import sys
import time
import asyncio
import threading
import io
import csv
from typing import Optional, List
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Query, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings
from database import db_manager
from pipeline import surveillance_pipeline
from recognition import face_recognizer
from registration import register_stakeholder
from utils.logger import get_logger

log = get_logger()

# Ensure directories exist
settings.ensure_directories()
db_manager.init_db()

VIDEO_UPLOAD_DIR = os.path.join(settings.DATA_DIR, "videos")
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Campus Sentinel - Surveillance & Stakeholder ID",
    version="2.0.0",
    description="Intelligent Real-time Campus Surveillance System"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Pipeline State Manager
# ---------------------------------------------------------------------------

class PipelineManager:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.source = settings.DEFAULT_SOURCE
        self.location = settings.DEFAULT_CAMERA_LOCATION
        self.client_queues: List[asyncio.Queue] = []
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.latest_telemetry = {
            "fps": 0.0,
            "summary": {"persons": 0, "recognized": 0, "unknown": 0},
            "events": [],
            "timestamp": time.time()
        }
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def register_client_queue(self, queue: asyncio.Queue):
        with self.lock:
            if queue not in self.client_queues:
                self.client_queues.append(queue)

    def unregister_client_queue(self, queue: asyncio.Queue):
        with self.lock:
            if queue in self.client_queues:
                self.client_queues.remove(queue)

    def frame_callback(self, frame, summary, events, fps):
        """Called by surveillance_pipeline on every rendered video frame."""
        # High quality & low latency JPEG encode
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, int(getattr(settings, "LIVE_JPEG_QUALITY", 80))])
        if not ok:
            return

        jpeg_bytes = encoded.tobytes()
        telemetry = {
            "fps": round(fps, 1),
            "summary": summary,
            "events": events,
            "timestamp": time.time()
        }

        with self.lock:
            self.latest_jpeg = jpeg_bytes
            self.latest_telemetry = telemetry
            queues = list(self.client_queues)

        # Distribute newest frame to all active async queues with drop-oldest guarantee
        if queues and self.loop and not self.loop.is_closed():
            for q in queues:
                def _push(queue=q, data=(jpeg_bytes, telemetry)):
                    if queue.full():
                        try:
                            queue.get_nowait()
                        except Exception:
                            pass
                    try:
                        queue.put_nowait(data)
                    except Exception:
                        pass
                try:
                    self.loop.call_soon_threadsafe(_push)
                except Exception:
                    pass

    def start(self, source=None, location=None):
        with self.lock:
            if self.is_running:
                return False, "Pipeline is already running."

            self.source = source if source is not None else settings.DEFAULT_SOURCE
            self.location = location or settings.DEFAULT_CAMERA_LOCATION
            self.stop_event.clear()
            self.is_running = True

            self.thread = threading.Thread(
                target=self._run_worker,
                args=(self.source, self.location),
                daemon=True
            )
            self.thread.start()
            log.info("Started surveillance pipeline on source %r @ %s", self.source, self.location)
            return True, "Pipeline started successfully."

    def _run_worker(self, source, location):
        try:
            surveillance_pipeline.run_surveillance(
                source=source,
                camera_location=location,
                display=False,
                frame_callback=self.frame_callback,
                stop_event=self.stop_event
            )
        except Exception as exc:
            log.exception("Pipeline worker encountered an error: %s", exc)
        finally:
            with self.lock:
                self.is_running = False
                self.latest_jpeg = None
            log.info("Surveillance worker terminated.")

    def stop(self):
        with self.lock:
            if not self.is_running:
                return False, "Pipeline is not running."
            self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        with self.lock:
            self.is_running = False
            self.thread = None
        log.info("Surveillance pipeline stopped.")
        return True, "Pipeline stopped."

    def get_status(self):
        with self.lock:
            return {
                "running": self.is_running,
                "source": str(self.source),
                "location": self.location,
                "fps": self.latest_telemetry["fps"],
                "summary": self.latest_telemetry["summary"],
                "ai_device": settings.AI_DEVICE,
            }


pipeline_mgr = PipelineManager()


@app.on_event("startup")
async def on_startup():
    pipeline_mgr.loop = asyncio.get_running_loop()
    log.info("Sentinel Web Server ready on port 8000. Access at http://localhost:8000")


@app.on_event("shutdown")
async def on_shutdown():
    pipeline_mgr.stop()


# ---------------------------------------------------------------------------
# WebSocket Real-Time Video & Event Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    Ultra-low latency WebSocket stream delivering binary JPEG frames
    and real-time detection telemetry at smooth 30+ FPS.
    """
    await websocket.accept()
    client_queue = asyncio.Queue(maxsize=1)
    pipeline_mgr.register_client_queue(client_queue)
    log.info("Live stream WebSocket client connected. Active streams: %d", len(pipeline_mgr.client_queues))

    # Send initial status
    try:
        await websocket.send_json({
            "type": "status",
            "data": pipeline_mgr.get_status()
        })
    except Exception:
        pass

    async def _send_frames():
        while True:
            jpeg_bytes, telemetry = await client_queue.get()
            await websocket.send_bytes(jpeg_bytes)
            if telemetry.get("events"):
                await websocket.send_json({"type": "events", "data": telemetry["events"]})

    async def _receive_client():
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"type": "pong", "time": time.time(), "status": pipeline_mgr.get_status()})

    sender_task = asyncio.create_task(_send_frames())
    receiver_task = asyncio.create_task(_receive_client())

    try:
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WebSocket live stream exception: %s", exc)
    finally:
        sender_task.cancel()
        receiver_task.cancel()
        pipeline_mgr.unregister_client_queue(client_queue)
        log.info("Live stream WebSocket client disconnected. Remaining: %d", len(pipeline_mgr.client_queues))


# ---------------------------------------------------------------------------
# Native MJPEG Video Stream
# ---------------------------------------------------------------------------

@app.get("/api/live/mjpeg")
async def api_mjpeg_stream():
    """Direct multipart MJPEG video stream for HTML5 <img src='/api/live/mjpeg'>."""
    async def mjpeg_generator():
        client_queue = asyncio.Queue(maxsize=1)
        pipeline_mgr.register_client_queue(client_queue)
        try:
            while True:
                jpeg_bytes, _ = await client_queue.get()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
        except asyncio.CancelledError:
            pass
        finally:
            pipeline_mgr.unregister_client_queue(client_queue)

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def api_get_stats():
    """Headline metrics for the dashboard."""
    stats = db_manager.get_stats()
    stats["pipeline"] = pipeline_mgr.get_status()
    return stats


@app.get("/api/pipeline/status")
def api_pipeline_status():
    """Current state of the AI surveillance pipeline."""
    return pipeline_mgr.get_status()


@app.post("/api/pipeline/start")
def api_pipeline_start(
    source: Optional[str] = Form(None),
    location: Optional[str] = Form(None)
):
    """Start or switch the surveillance pipeline source."""
    resolved_source = source
    if source is not None and str(source).strip().isdigit():
        resolved_source = int(source)

    success, message = pipeline_mgr.start(source=resolved_source, location=location)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message, "pipeline": pipeline_mgr.get_status()}


@app.post("/api/pipeline/stop")
def api_pipeline_stop():
    """Stop the running surveillance pipeline."""
    success, message = pipeline_mgr.stop()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message, "pipeline": pipeline_mgr.get_status()}


@app.get("/api/visits")
def api_get_visits(
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = None,
    role: Optional[str] = None
):
    """Fetch paginated visit logs."""
    rows = db_manager.fetch_visits(limit=limit, offset=offset, name_filter=search, role_filter=role)
    visits = []
    for r in rows:
        visits.append({
            "id": r[0],
            "timestamp": r[1],
            "name": r[2],
            "role": r[3],
            "uid": r[4],
            "location": r[5],
            "similarity": r[6],
            "image_path": f"/data/stakeholders/{os.path.basename(r[7])}" if r[7] and os.path.exists(r[7]) else None
        })
    return {"visits": visits, "count": len(visits)}


@app.get("/api/visits/export")
def api_export_visits(search: Optional[str] = None, role: Optional[str] = None):
    """Export visit records as a downloadable CSV."""
    rows = db_manager.fetch_visits(limit=10000, offset=0, name_filter=search, role_filter=role)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Name", "Role", "UID", "Camera Location", "Similarity"])
    for r in rows:
        writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6]])

    output.seek(0)
    filename = f"visit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/unknowns")
def api_get_unknowns(
    limit: int = 100,
    offset: int = 0,
    only_unverified: bool = False
):
    """Fetch unknown person records."""
    rows = db_manager.fetch_unknowns(limit=limit, offset=offset, only_unverified=only_unverified)
    unknowns = []
    for r in rows:
        img_url = f"/data/unknown_faces/{os.path.basename(r[3])}" if r[3] and os.path.exists(r[3]) else None
        unknowns.append({
            "id": r[0],
            "timestamp": r[1],
            "location": r[2],
            "image_url": img_url,
            "raw_image_path": r[3],
            "verified": bool(r[4])
        })
    return {"unknowns": unknowns, "count": len(unknowns)}


@app.post("/api/unknowns/{unknown_id}/verify")
def api_verify_unknown(unknown_id: int):
    """Mark an unknown capture record as verified."""
    db_manager.mark_unknown_verified(unknown_id)
    return {"status": "ok", "id": unknown_id, "verified": True}


@app.delete("/api/unknowns/{unknown_id}")
def api_delete_unknown(unknown_id: int):
    """Delete an unknown capture record."""
    db_manager.delete_unknown_person(unknown_id)
    return {"status": "ok", "id": unknown_id, "deleted": True}


@app.get("/api/stakeholders")
def api_get_stakeholders():
    """List all registered campus stakeholders."""
    rows = db_manager.list_stakeholders()
    stakeholders = []
    for r in rows:
        img_url = f"/data/stakeholders/{os.path.basename(r[4])}" if r[4] and os.path.exists(r[4]) else None
        stakeholders.append({
            "id": r[0],
            "uid": r[1],
            "name": r[2],
            "role": r[3],
            "image_url": img_url,
            "registered_at": r[5]
        })
    return {"stakeholders": stakeholders, "count": len(stakeholders)}


@app.post("/api/stakeholders/register")
async def api_register_stakeholder(
    uid: str = Form(...),
    name: str = Form(...),
    role: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """Enroll a new stakeholder via uploaded photo file."""
    if not file:
        raise HTTPException(status_code=400, detail="Please provide a reference photo.")

    # Save uploaded image to temp directory
    temp_dir = os.path.join(settings.DATA_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uid}_{file.filename}")

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        sid = register_stakeholder.register_from_images(uid.strip(), name.strip(), role.strip(), temp_path)
        if sid is None:
            raise HTTPException(
                status_code=400,
                detail="No clear face could be detected in the uploaded photo. Please upload a clear front-facing portrait."
            )

        # Trigger immediate gallery reload in running pipeline
        surveillance_pipeline.refresh_gallery(force=True)

        return {
            "status": "ok",
            "message": f"Successfully enrolled {name} ({role}) with UID {uid}",
            "id": sid,
            "uid": uid
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.delete("/api/stakeholders/{uid}")
def api_delete_stakeholder(uid: str):
    """Delete a stakeholder by UID."""
    db_manager.delete_stakeholder(uid)
    surveillance_pipeline.refresh_gallery(force=True)
    return {"status": "ok", "uid": uid, "deleted": True}


@app.get("/api/reports")
def api_get_reports():
    """Aggregated timeline and categorical analytics for charts."""
    return db_manager.get_reports_data()


@app.post("/api/videos/upload")
async def api_upload_video(file: UploadFile = File(...)):
    """Upload a recorded video file for surveillance evaluation."""
    filename = file.filename
    save_path = os.path.join(VIDEO_UPLOAD_DIR, filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    return {"status": "ok", "filename": filename, "path": save_path}


@app.get("/api/videos")
def api_list_videos():
    """List available recorded video files."""
    if not os.path.exists(VIDEO_UPLOAD_DIR):
        return {"videos": []}
    files = [f for f in os.listdir(VIDEO_UPLOAD_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))]
    return {"videos": sorted(files)}


# ---------------------------------------------------------------------------
# Static File Mounts & Frontend Single Page App
# ---------------------------------------------------------------------------

# Static data mounts
app.mount("/data/stakeholders", StaticFiles(directory=settings.STAKEHOLDER_IMG_DIR), name="stakeholders_data")
app.mount("/data/unknown_faces", StaticFiles(directory=settings.UNKNOWN_IMG_DIR), name="unknowns_data")
app.mount("/data/live", StaticFiles(directory=settings.LIVE_FRAME_DIR), name="live_data")

WEB_DIR = os.path.join(PROJECT_ROOT, "web")
os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(os.path.join(WEB_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(WEB_DIR, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the modern single-page dashboard."""
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Campus Sentinel Web UI loading...</h2>", status_code=200)


def start_server(host="0.0.0.0", port=8000):
    """Launch the uvicorn web server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
