#  Real-Time Campus Stakeholder Identification and Unknown Person Registration

An AI-powered intelligent campus surveillance system that performs **real-time person detection**, **stakeholder identification**, and **unknown person registration** using **YOLOv8**, **InsightFace**, **OpenCV**, **SQLite**, and **Streamlit**.

---

##  Project Overview

Traditional CCTV systems rely on continuous manual monitoring, making it difficult to efficiently identify campus stakeholders and monitor unauthorized individuals. This project aims to automate the surveillance process by leveraging state-of-the-art computer vision and deep learning models.

The system detects people from live CCTV feeds, recognizes registered stakeholders through face recognition, and automatically records unknown individuals for future verification. A Streamlit dashboard provides real-time monitoring, analytics, and search capabilities.

---

##  Features

* Real-time CCTV/Webcam monitoring
* Person detection using YOLOv8
* Face detection and embedding extraction using InsightFace
* Stakeholder identification using cosine similarity
* Automatic visit logging
* Unknown person registration
* SQLite database for centralized data management
* Interactive Streamlit dashboard
* Visit analytics and reports
* Search stakeholder history

---

# Instructions to run the project:

-  1. check registration(preview, without touching anything): `python registration/bulk_register.py --sheet stakeholders.xlsx --photos-dir stakeholder --dry-run`

-  2. register for real: `python registration/bulk_register.py --sheet stakeholders.xlsx --photos-dir stakeholder`

-  3. start survellience: `python main.py run --location "MBUST Lab"`

-  4. start the dashboard: `streamlit run dashboard/app.py`


## 📁 Project structure

```
campus_surveillance/
├── main.py                        # CLI: init-db | check | register | run | list
├── requirements.txt
├── config/settings.py             # every path & threshold in one place
├── camera/webcam_stream.py        # laptop webcam (auto-detects index 0/1/2)
├── detection/person_detector.py   # YOLOv8 person detection
├── recognition/face_recognizer.py # InsightFace embeddings + cosine matching
├── database/db_manager.py         # SQLite: stakeholders, visit_logs, unknown_persons
├── pipeline/surveillance_pipeline.py  # real-time loop
├── registration/register_stakeholder.py  # enroll via webcam or photos
├── dashboard/app.py               # Streamlit dashboard
├── utils/                         # logger + image helpers
├── models/                        # (optional) fine-tuned YOLO weights
└── data/                          # created automatically at runtime
    ├── campus_surveillance.db
    ├── stakeholders/
    ├── unknown_faces/
    └── live/
```

## ❓ What goes in each data folder? (short answer: nothing — leave them empty)

| Folder | Who fills it | What ends up inside |
|---|---|---|
| `data/stakeholders/` | **The program**, during registration | One reference photo per registered person, named by UID (e.g. `S001.jpg`). Shown in the dashboard. **Don't put files here manually.** |
| `data/unknown_faces/` | **The program**, during surveillance | Auto-cropped face images of unrecognized people, timestamped (e.g. `unknown_20260704_101502_123456.jpg`). Review them in the dashboard's *Unknown Persons* tab. |
| `data/live/` | **The program**, during surveillance | A single `latest.jpg` — the most recent annotated frame — which the dashboard's *Live Monitor* tab displays. Overwritten continuously. |
| `data/campus_surveillance.db` | **The program** | The SQLite database (stakeholders, visit logs, unknown records). |

The only folder **you** create is an optional `photos/` folder (anywhere you
like) if you prefer registering people from photo files instead of the
webcam — e.g. `photos/sita/1.jpg, 2.jpg, 3.jpg` (3–5 clear, front-facing
photos per person give the best accuracy).

---

## 🚀 Complete workflow — follow these steps in order

### Step 0 — Prerequisites (one time)
* Python **3.9 – 3.11** installed (`python --version`)
* Internet connection for the **first run only** (models auto-download:
  `yolov8n.pt` ≈ 6 MB and InsightFace `buffalo_l` ≈ 280 MB)
* Webcam not in use by Zoom/Teams/browser, and OS camera permission granted
  to your terminal (Windows: Settings → Privacy → Camera; macOS: System
  Settings → Privacy & Security → Camera)

### Step 1 — Install (one time)
```bash
cd sentinel
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Initialize the database (one time)
```bash
python main.py init-db
```
Creates `data/` and all its subfolders + the SQLite database.

### Step 3 — Self-check (one time, strongly recommended)
```bash
python main.py check
```
Verifies every library, downloads/loads both models, initializes the DB and
tests your webcam. Fix any `[FAIL]` line before continuing — if this prints
**ALL CHECKS PASSED**, the pipeline will run without setup errors.

### Step 4 — Register stakeholders (once per person)
Easiest — straight from the webcam (captures 5 face samples automatically;
look at the camera, turn your head slightly between captures):
```bash
python main.py register --uid S001 --name "Sita Sharma" --role Student --webcam
```
Or from photo files (file, folder, or glob; multiple photos are averaged
into one robust template):
```bash
python main.py register --uid F010 --name "Dr. Ram K." --role Faculty --images ./photos/ram/
```
Roles: `Student`, `Faculty`, `Staff`, `Authorized`.
Re-registering the same `--uid` safely **updates** that person.

### Step 5 — Confirm registration
```bash
python main.py list
```

### Step 6 — Run real-time surveillance
```bash
python main.py run
```
* A preview window opens: **green box** = recognized stakeholder (name, role,
  similarity), **red box** = UNKNOWN (face auto-saved + DB record).
* Visits are logged at most once per person per 60 s; the same unknown is
  not re-saved within 120 s (both tunable in `config/settings.py`).
* Press **q** in the preview window (or Ctrl-C in the terminal) to stop.

Useful variants:
```bash
python main.py run --location "Home Desk"     # label stored with every log
python main.py run --source 1                 # external USB webcam
python main.py run --source test.mp4 --max-frames 300   # evaluate on a video
python main.py run --no-display               # headless (no preview window)
```

### Step 7 — Open the dashboard (second terminal, same venv)
```bash
streamlit run dashboard/app.py
```
Tabs: **Live Monitor** (latest annotated frame) • **Visit Logs** (search +
CSV export) • **Unknown Persons** (photo review queue with *Mark verified*)
• **Stakeholders** • **Reports** (visits per day / location / role).

### Everyday use after setup
```bash
source .venv/bin/activate        # (or .venv\Scripts\activate)
python main.py run               # terminal 1
streamlit run dashboard/app.py   # terminal 2
```

---

## 🔧 Tuning (config/settings.py)

| Setting | Default | Effect |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | 0.45 | Lower ⇒ easier match (more false accepts); higher ⇒ stricter (more false UNKNOWNs). Try 0.40–0.50. |
| `VISIT_LOG_COOLDOWN_SEC` | 60 | Gap between repeat visit logs per person |
| `UNKNOWN_LOG_COOLDOWN_SEC` | 120 | De-dup window for unknown captures |
| `FRAME_PROCESS_EVERY_N` | 2 | Raise to 3–4 on slow laptops for smoother FPS |
| `DISPLAY_WINDOW` | True | Set False on headless machines |

## 🩹 Troubleshooting

| Symptom | Fix |
|---|---|
| `No working webcam found` | Close apps using the camera; grant camera permission to the terminal; try `--source 1`. |
| First run is very slow | Models are downloading (one time). Watch the progress in the terminal. |
| Everyone shows as UNKNOWN | You skipped Step 4 — the log even warns "gallery is EMPTY". Register people first. |
| You show as UNKNOWN despite being registered | Improve lighting, face the camera, re-register with more/better samples, or lower `FACE_MATCH_THRESHOLD` slightly. |
| Preview window frozen / won't close | Click the window, press `q`; or Ctrl-C in the terminal. |
| Dashboard shows "No live frame yet" | Start `python main.py run` first, then click 🔄 Refresh. |

---

# Installation Guide

## Clone Repository

```bash
git clone https://github.com/dilliram-code/sentinel.git

cd sentinel
```
## Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
```

---

## Activate Environment

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Main Detection System

```bash
python main.py run --location "MBUST Lab"
```

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```
---

## Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
```

---

## Activate Environment

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Main Detection System

```bash
python main.py
```

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Development Phases 📈

## Phase 1

* Project setup
* Folder structure
* Webcam testing
* YOLO integration

---

## Phase 2

* SQLite database
* Streamlit dashboard
* Stakeholder management
* Unknown person logging

---

# 📈 Future Enhancements

* Multi-camera synchronization
* Person tracking using ByteTrack
* Automatic attendance management
* Visitor management system
* Face search from uploaded image
* REST API backend
* Docker deployment
* Cloud database support
* Role-based authentication
* Heatmap visualization
* Campus movement analytics
* Mobile application

---

# Sample Workflow

```text
Camera
   │
   ▼
YOLOv8
   │
   ▼
Detect Person
   │
   ▼
Extract Face
   │
   ▼
InsightFace
   │
   ▼
Embedding
   │
   ▼
SQLite Database
   │
   ▼
Known?──────────►YES────────►Visit Logged
   │
   ▼
NO
   │
   ▼
Save Unknown Face
   │
   ▼
Dashboard Updated
```

---

# Learning Objectives

This project demonstrates practical implementation of:

* Computer Vision
* Deep Learning
* Face Recognition
* Object Detection
* Functional Programming
* Database Design
* Streamlit Application Development
* Real-Time Video Processing
* AI System Integration

---

# Contributors

* **Dilli Ram Chaudhary** (Master of Artificial Intelligence)
* **Piyush Lal Shrestha** (Master of Data Science)
* **Yalamber Ingnam** (Master of Data Science)

---
# ⚖️ Ethics note
Even in a webcam demo, face embeddings are biometric data: enroll only
people who consent, and delete `data/` when the demonstration is finished.
# License

This project is developed for academic and research purposes.

---
⭐ If you find this project helpful, consider giving it a star on GitHub😊!


