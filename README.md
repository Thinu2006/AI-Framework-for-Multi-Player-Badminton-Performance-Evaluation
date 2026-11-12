# Badminton AI Vision & Biomechanical Coaching System

An end-to-end Computer Vision and Retrieval-Augmented Generation (RAG) AI system designed for badminton match video analysis, player tracking, stroke classification, biomechanical flaw detection, and automated AI coaching advice synthesis.

---

## 🏗️ System Architecture

The application operates as a full-stack AI processing platform:

1. **Vite + React Dashboard (`frontend/`)**: Interface for coaches to manage players, upload match videos, monitor real-time 7-stage processing progress, and view interactive biomechanical diagnostic reports.
2. **Flask REST API (`flask_api.py`)**: Asynchronous REST backend handling JWT authentication, video upload validation, background worker thread queueing, and report serialization.
3. **Multi-Stage Computer Vision & AI Pipeline (`main_pipeline.py`)**: Core processing engine combining YOLOv8, MoveNet, TrackNet, custom homography geometry, biomechanical error estimators, and OpenRouter/OpenAI LLMs.
4. **PostgreSQL / SQLite Database Adapter (`database.py`)**: Centralized storage for coach accounts, player profiles, session metadata, job statuses, and participant configurations.

```
       +-------------------------------------------------------------+
       |                    Client Browser                           |
       |             (React / Vite Single Page App)                  |
       +------------------------------+------------------------------+
                                      | HTTP / REST (JWT Auth)
                                      v
       +-------------------------------------------------------------+
       |                  Flask REST API Backend                     |
       |                (flask_api.py / Thread Workers)              |
       +--------------+-------------------------------+--------------+
                      |                               |
                      v                               v
       +------------------------------+ +----------------------------+
       | PostgreSQL / SQLite Database | | 7-Stage Multi-AI Pipeline  |
       | (Coaches, Sessions, Players) | | (YOLO, MoveNet, TrackNet,  |
       +------------------------------+ |  RAG LLM Coaching Generator|
                                        +----------------------------+
```

---

## 🔬 7-Stage AI & CV Pipeline

The analysis pipeline processes raw video input through 7 sequential stages:

* **Stage 1: Court & Tactical Zone Calibration**
  * `AutoCourtDetector` automatically identifies court corner pixels.
  * `CourtMapper` & `CourtGeometry` compute inverse Homography matrices to transform perspective video space into a standardized 2D metric court grid (6 tactical zones per side).
* **Stage 2: Player Detection & Role Assignment**
  * `YOLOv8` detects players on court and assigns roles (**Near Player** vs **Far Player**) based on court polygon boundaries and spatial orientation.
* **Stage 3: Body Posture Estimation**
  * `MoveNet Thunder` (TFLite) estimates 17 2D keypoints per player frame-by-frame, providing precise ankle-based foot placement on the court plane and skeletal joint tracking.
* **Stage 4: Shuttle Flight Tracking**
  * `TrackNet` (ResNet architecture) tracks the shuttlecock across video frames, mapping pixel trajectories into 2D metric court coordinates.
* **Stage 5: Shot Segmentation & Stroke Recognition**
  * `BadmintonShotDetector` segments rally shots, identifying stroke types (*Smash, Clear, Drop, Lift, Net Shot, Drive, Serve*), hitter identity, travel directions, and frame ranges.
* **Stage 6: Technique & Biomechanical Flaw Analysis**
  * `error_detection/stage6_error_pipeline.py` analyzes contact points, phase segmentation (preparation, contact, follow-through), joint angles (elbow, shoulder, knee, hip), and flags biomechanical errors against `shot_rule_repository.json`.
* **Stage 7: Grounded AI Coach Advice Synthesis**
  * `RAG/stage7_feedback_pipeline.py` matches detected errors against the official **Sri Lanka Badminton Association (SLBA) Level-1 Coach Education** reference rules, utilizing an LLM (OpenRouter / OpenAI) to generate tailored corrective drills and personalized feedback.

---

## 🛠️ Tech Stack & Models

* **Core Language**: Python 3.10+
* **Computer Vision**: OpenCV (`cv2`), NumPy, Scikit-Learn
* **Deep Learning Frameworks**: TensorFlow 2.x, PyTorch (`ultralytics`), TFLite
* **Pre-trained AI Models**:
  * `yolov8n.pt` — Player detection
  * `movenet_thunder.tflite` — High-precision pose keypoint estimation
  * `TrackNet` — Shuttlecock trajectory tracking
* **RAG & LLM Synthesis**: PyPDF (`pypdf`), OpenAI SDK (`openai`), OpenRouter API
* **Web API**: Flask 3.0, Flask-CORS, PyJWT
* **Database**: PostgreSQL (`psycopg2`) / SQLite (`badminton.db` fallback)
* **Frontend**: React 18, Vite, Tailwind CSS / Lucide icons

---

## 📁 Directory Structure

```
E:\new_court_detection/
├── court_detection/            # Stage 1: Auto court detection & homography mapping
│   ├── auto_detector.py        # Automatic court corner localization
│   ├── court_geometry.py       # 2D metric court coordinate transformations
│   └── court_mapper.py         # Bird's-eye view renderer & overlay builder
├── player_detection/           # Stage 2: Player bounding box detection & role assignment
│   ├── detect_player.py        # YOLOv8 player tracking filter
│   └── yolov8n.pt              # YOLOv8 network weights
├── pose_estimate/              # Stage 3: MoveNet pose keypoint estimation
│   ├── pose_estimation.py      # TFLite MoveNet Thunder pose estimator
│   └── movenet_thunder.tflite  # MoveNet model file
├── shuttle_detection/          # Stage 4: TrackNet shuttlecock tracking
│   ├── TrackNet.py             # ResNet TrackNet network architecture
│   └── predict.py              # Shuttle trajectory prediction script
├── shot_detection/             # Stage 5: Badminton stroke classification
│   ├── shot_classification.py  # Rule-based & heuristic stroke classifier
│   └── shot_segmentation.py    # Rally stroke segmentation pipeline
├── error_detection/            # Stage 6: Biomechanical joint angle flaw analysis
│   ├── stage6_error_pipeline.py# Joint angle feature extraction & rule validation
│   ├── contact_estimator.py    # Impact frame detection
│   └── shot_rule_repository.json # SLBA technical rule repository
├── RAG/                        # Stage 7: PDF rule extraction & LLM coach advice
│   ├── SLBA COACH EDUCATION LEVEL-1.pdf # Technical standard reference manual
│   ├── shot_rag_extractor.py   # PDF rule indexing & embedding retriever
│   └── stage7_feedback_pipeline.py  # LLM feedback generator
├── frontend/                   # React + Vite web dashboard application
├── testing/                    # Test mode evaluation modules & benchmark runner
├── csv_exporter.py             # Export player statistics & court coordinate CSVs
├── database.py                 # Dual PostgreSQL / SQLite DB adapter
├── flask_api.py                # REST API backend server & job manager
├── llm_config.json             # LLM provider configuration & API key
├── llm_prompts.py              # System & coaching prompt templates
├── main_pipeline.py            # Master 7-stage pipeline orchestrator
├── pipeline_logger.py          # Detailed event logging engine
├── report_view_builder.py      # Aggregates pipeline output into JSON reports
└── requirements.txt            # Python dependencies
```

---

## 💻 Getting Started & Setup

### Prerequisites
* Python 3.10+
* Node.js 18+ & npm

### 1. Python Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Backend Flask API Server
```bash
python flask_api.py
```
The REST API server will start on `http://127.0.0.1:5000`. If PostgreSQL is not configured, it automatically uses the local SQLite database (`badminton.db`).

### 3. Run Frontend Web Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 🔗 API Reference

### Authentication
* `POST /api/register` — Register a new coach account.

* `POST /api/login` — Authenticate coach credentials and receive JWT bearer token.

### Player & Session Management
* `GET /api/players` — Get all registered players for the logged-in coach.
* `POST /api/players` — Register a new player profile.
* `DELETE /api/players/<player_id>` — Delete a player profile.
* `GET /api/sessions` — Fetch list of analyzed match sessions.
* `DELETE /api/sessions/<session_id>` — Delete session and associated reports.

### Video Analysis & Reports
* `POST /api/analyze` — Upload match video (`multipart/form-data`) to trigger background 7-stage processing.
* `GET /api/analyze/status/<job_id>` — Query live percentage and stage progress for a running job.
* `POST /api/analyze/cancel/<job_id>` — Cancel an in-progress analysis job.
* `GET /api/report/<session_id>` — Fetch structured biomechanical report, shot predictions, errors, and coaching feedback.
* `GET /api/test_report/<session_id>` — Fetch test evaluation benchmark reports against ground truth.

---

## ⚙️ Configuration

### LLM Provider (`llm_config.json`)
Configure your LLM model provider (OpenRouter / OpenAI) for Stage 7 feedback generation:
```json
{
  "api_key": "sk-or-v1-xxxxxxxxxxxxxxxx",
  "model": "openai/gpt-4o-mini",
  "provider": "openrouter",
  "base_url": "https://openrouter.ai/api/v1"
}
```

### Shot Rules (`error_detection/shot_rule_repository.json`)
Contains biomechanical rule bounds (elbow angle, racket preparation posture, foot placement) derived from the SLBA manual for each stroke type (*Smash, Drop, Clear, Drive, Net Shot*).

---

## 📄 License & Acknowledgments

* **Technical Standards**: Sri Lanka Badminton Association (SLBA) Level-1 Coach Education Manual.
* **Computer Vision Architectures**: YOLOv8 (Ultralytics), MoveNet (Google TensorFlow), TrackNet.
