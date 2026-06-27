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

## Project Architecture

```text
CampusAIStakeholderDetection/

├── config/
├── dashboard/
├── database/
├── data/
│   ├── stakeholders/
│   └── unknown_faces/
├── models/
├── services/
├── utils/
├── tests/
├── reports/
├── main.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

| Category             | Technology        |
| -------------------- | ----------------- |
| Programming Language | Python            |
| Person Detection     | YOLOv8            |
| Face Recognition     | InsightFace       |
| Face Similarity      | Cosine Similarity |
| Computer Vision      | OpenCV            |
| Database             | SQLite            |
| Dashboard            | Streamlit         |
| Deep Learning        | PyTorch           |
| Data Processing      | NumPy, Pandas     |
| Visualization        | Plotly            |

---

# 🔄 Project Workflow

```text
                Live CCTV Camera
                      │
                      ▼
             Video Frame Capture
               (OpenCV VideoCapture)
                      │
                      ▼
         YOLOv8 Person Detection Model
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
 No Person                       Person Detected
                                      │
                                      ▼
                            Crop Detected Person
                                      │
                                      ▼
                           Face Detection Module
                                      │
                          Face Found?
                        ┌──────┴───────┐
                        ▼              ▼
                      No Face        Face Found
                        │              │
                        ▼              ▼
                  Ignore Frame   InsightFace Model
                                      │
                                      ▼
                          Face Embedding Extraction
                                      │
                                      ▼
                     Compare with Stakeholder Database
                         (Cosine Similarity Matching)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
         Stakeholder Recognized                  Unknown Person
                 │                                         │
                 ▼                                         ▼
         Log Visit Record                    Save Face Image
                 │                                         │
                 ▼                                         ▼
           SQLite Database                  Unknown Person Database
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                           Streamlit Dashboard
                                      │
          ┌───────────────┬──────────────┬───────────────┐
          ▼               ▼              ▼               ▼
     Live Feed     Visit History   Unknown Persons   Analytics
```

---

# Recognition Pipeline

```text
Frame
  │
  ▼
YOLOv8
(Person Detection)
  │
  ▼
Crop Person
  │
  ▼
InsightFace
(Face Detection)
  │
  ▼
Face Embedding (512-D Vector)
  │
  ▼
Cosine Similarity
  │
  ▼
Known? ──────────────► Yes ─► Log Visit
  │
  ▼
No
  │
  ▼
Save Unknown Face
```

---

# 📊 Database Schema

## Stakeholders

| Column         | Description           |
| -------------- | --------------------- |
| stakeholder_id | Primary Key           |
| full_name      | Stakeholder Name      |
| role           | Student/Faculty/Staff |
| department     | Department            |
| image_path     | Stored Image          |
| embedding      | Face Embedding        |

---

## Visits

| Column          | Description |
| --------------- | ----------- |
| visit_id        | Primary Key |
| stakeholder_id  | Foreign Key |
| camera_location | Camera Name |
| timestamp       | Visit Time  |

---

## Unknown Persons

| Column          | Description    |
| --------------- | -------------- |
| unknown_id      | Primary Key    |
| image_path      | Captured Face  |
| camera_location | Camera Name    |
| timestamp       | Detection Time |

---

# Streamlit Dashboard

The dashboard provides the following modules:

* Live Monitoring
* Stakeholder Management
* Visit History
* Unknown Person Gallery
* Analytics Dashboard
* Search Records
* Reports

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/CampusAIStakeholderDetection.git

cd CampusAIStakeholderDetection
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

# Development Phases

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

# License

This project is developed for academic and research purposes.

---

⭐ If you find this project helpful, consider giving it a star on GitHub!
