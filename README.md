# ⚽ Sports Player & Ball Tracking Analytics

> **CSE 429 · Computer Vision and Pattern Recognition · E-JUST**  
> Multi-Object Tracking (MOT) system for sports video analysis using **ByteTrack**, **YOLOv8**, and **Streamlit**.

---

# RTX3060
## CUDA-enabled PyTorch (WSL2)

If CUDA is not detected, reinstall PyTorch with the CUDA 12.4 wheels:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Configuration (.env)](#-configuration-env)
- [Pipeline Explained](#-pipeline-explained)
- [Module Reference](#-module-reference)
- [Tracker Comparison](#-tracker-comparison)
- [Known Issues & Notes](#-known-issues--notes)
- [Dependencies](#-dependencies)

---

## 🔭 Overview

This project implements a full **sports video analytics pipeline** that:

1. **Detects** players using YOLOv8m (with SAHI slicing for panoramic cameras)
2. **Tracks** them frame-by-frame using ByteTrack (two-pass matching)
3. **Detects the ball** using a dedicated ball detection model + Kalman gate filter
4. **Classifies teams** via unsupervised color clustering (RGB + HSV)
5. **Computes ball possession** based on player proximity
6. **Compares trackers** (ByteTrack vs SORT vs SORT+ReID)
7. **Visualises** everything in an interactive Streamlit dashboard

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Player Detection** | YOLOv8m with optional SAHI slicing for wide/panoramic cameras |
| 🏃 **ByteTrack** | Two-pass tracker — rescues low-confidence detections, fewer ID switches |
| ⚽ **Ball Detection** | Dedicated football detection model + Kalman gate ghost-dot rejection |
| 👕 **Team Classification** | KMeans clustering on jersey color (RGB + HSV histogram features) |
| 📊 **Possession Analysis** | Ball-to-player proximity across time |
| 🔥 **Heatmaps** | KDE-based player position density maps per team |
| 📹 **Frame Preview** | Annotated frame-by-frame preview with bounding boxes |
| 🎬 **Video Export** | Render and download fully annotated video |
| 🔄 **SORT Comparison** | Side-by-side comparison of ByteTrack vs SORT vs SORT+ReID |
| 🔺 **Defensive Triangle** | Optional pitch homography overlay (requires calibration) |

---

## 🏗 Architecture

```
Video Input
    │
    ▼
┌─────────────────────┐
│  PlayerDetector      │  YOLOv8m (+ SAHI for panoramic)
│  src/detection.py    │
└────────┬────────────┘
         │ sv.Detections
         ▼
┌─────────────────────┐
│  ByteTrackWrapper    │  Two-pass matching
│  src/tracking.py     │  High-conf → Low-conf rescue
└────────┬────────────┘
         │ tracks_df (DataFrame)
         ▼
┌──────────────────────────────────────────────────┐
│  Parallel pipelines                              │
│                                                  │
│  BallDetector ──────────────► ball_df            │
│  src/ball.py   Kalman gate                       │
│                                                  │
│  classify_teams ────────────► tracks_df + team   │
│  src/teams.py  KMeans color                      │
│                                                  │
│  compute_possession ────────► poss_df            │
│  src/possession.py                               │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit Dashboard │  app.py
│  Tabs: Preview /     │
│  Possession /        │
│  Heatmaps /          │
│  Ball Trajectory /   │
│  ByteTrack vs SORT   │
└─────────────────────┘
```

---

## 📁 Project Structure

```
MOT-main/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── config.py               # Global configuration (reads .env)
├── .env                    # Environment overrides (device, confidence, etc.)
│
├── src/                    # Core pipeline modules
│   ├── __init__.py
│   ├── detection.py        # PlayerDetector — YOLOv8 + SAHI
│   ├── tracking.py         # ByteTrackWrapper, SORT, SORT+ReID, Kalman tracker
│   ├── ball.py             # BallDetector — dedicated model + Kalman gate
│   ├── teams.py            # Team colour clustering (KMeans)
│   ├── possession.py       # Ball possession computation
│   ├── heatmap.py          # Heatmaps, ball trajectory, possession chart
│   ├── homography.py       # Pitch homography (camera calibration)
│   ├── video.py            # Annotated video rendering
│   ├── evaluation.py       # Tracker comparison metrics & charts
│   └── overlays.py         # Defensive triangle drawing helpers
│
├── models/                 # Downloaded model weights (auto-created)
│   └── ball_detector.pt    # Downloaded on first run from HuggingFace
│
├── outputs/                # All generated outputs (auto-created)
│   ├── videos/             # Rendered annotated videos
│   ├── stats/              # CSV exports (tracks, ball, possession, sort)
│   └── heatmaps/           # Heatmap images
│
├── img1.mp4                # Bundled demo video
├── yolov8m.pt              # Player detection model (YOLOv8 medium)
├── yolov8x.pt              # Ball detection fallback model (YOLOv8 extra-large)
│
├── requirements.txt        # Python dependencies
├── smoke_test.py           # Quick import & config sanity check
├── calibrate.py            # Pitch homography calibration helper
├── codes/                  # Additional evaluation scripts
└── .streamlit/
    └── config.toml         # Streamlit theme config
```

---

## 🚀 Installation

### Prerequisites

- Python **3.10 – 3.12**
- pip ≥ 24
- (Optional) NVIDIA GPU with CUDA 12.4 for faster inference

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/Bytrack_MOT_main/MOT-main
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv

# Activate (Linux / macOS / WSL)
source .venv/bin/activate

# Activate (Windows CMD)
.venv\Scripts\activate.bat
```

### Step 3 — Install PyTorch

**With CUDA (recommended for GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Verify installation

```bash
python smoke_test.py
```

Expected output:
```
✅ torch X.X.X  | CUDA: True/False
✅ ultralytics OK
✅ supervision X.X.X
✅ streamlit X.X.X
✅ All checks passed
```

---

## ▶️ Running the App

```bash
# Make sure you're in the MOT-main directory with .venv active
.venv/bin/streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **⚠️ WSL2 / Windows users:** First boot is slow (~2–5 minutes) because Python
> imports large ML libraries (torch, ultralytics) across the NTFS filesystem boundary.
> The page will appear blank/skeleton during this time — this is normal.
> Once the server prints `Uvicorn server started on 0.0.0.0:8501`, refresh your browser.

### Quick start with the demo video

1. Open http://localhost:8501
2. Check **"Or use bundled demo video (img1.mp4)"**
3. Click **🚀 Run Analysis**
4. Wait for all 5 steps to complete
5. Explore the result tabs

---

## ⚙️ Configuration (.env)

The `.env` file in the project root controls all runtime parameters. Copy and edit as needed:

```ini
# Device: "auto" (GPU if available, else CPU), "cuda", or "cpu"
SPORTSMOT_DEVICE=auto

# Player detection model filename (must be in project root)
SPORTSMOT_PLAYER_MODEL_NAME=yolov8m.pt

# Detection confidence thresholds
SPORTSMOT_DET_CONF_PANORAMIC=0.15    # Lower — players appear smaller
SPORTSMOT_DET_CONF_BROADCAST=0.35    # Higher — closer camera

# ByteTrack two-pass split threshold
SPORTSMOT_BYTETRACK_HIGH_CONF_PANORAMIC=0.35
SPORTSMOT_BYTETRACK_HIGH_CONF_BROADCAST=0.45

# Upload / processing limits
SPORTSMOT_MAX_UPLOAD_MB=800
SPORTSMOT_MAX_VIDEO_MINUTES=20
```

All values have sensible defaults in `config.py` and the `.env` is optional.

---

## 🔬 Pipeline Explained

### 1. Video Mode Detection

The app automatically detects whether the video is **panoramic** (aspect ratio > 3.0) or **broadcast**:
- Panoramic → lower confidence thresholds + SAHI slicing (players are tiny)
- Broadcast → standard direct YOLO detection

### 2. Player Detection (`src/detection.py`)

- Uses **YOLOv8m** with person class only (`class_id=0`)
- For panoramic: uses **SAHI** (Sliced Aided Hyper Inference) — divides the frame into 540×540 overlapping tiles, runs detection on each, merges results
- Returns `supervision.Detections` objects

### 3. ByteTrack (`src/tracking.py`)

- **Two-pass matching strategy:**
  1. High-confidence detections (≥ threshold) matched first via IoU
  2. Low-confidence detections get a "rescue pass" against lost tracks
- Maintains a **lost track buffer** (30 frames) — players hidden behind other players stay tracked
- SORT is run on the same detections in parallel for comparison

### 4. Ball Detection (`src/ball.py`)

- Downloads a dedicated football detection model from HuggingFace on first run
- Falls back to YOLOv8x COCO "sports ball" class if download fails
- **BallKalmanGate**: A Kalman filter predicts ball position each frame. Detections that are > 120px from the predicted position are rejected as "ghost dots" (reflection artefacts, crowd noise)

### 5. Team Classification (`src/teams.py`)

- Crops the torso region of each detected player
- Computes RGB + HSV colour histograms (16 bins each)
- Runs **KMeans (k=2)** on all tracks
- Handles overlapping players with IoU-based duplicate suppression

### 6. Possession Analysis (`src/possession.py`)

- For each frame, finds the player closest to the ball
- Assigns possession if distance < 150px
- Accumulates possession time per team across all frames

### 7. Evaluation (`src/evaluation.py`)

Compares ByteTrack, SORT, and SORT+ReID on:
- Total unique IDs (fewer = better track consistency)
- ID switches per frame
- Track fragmentation

---

## 📦 Module Reference

### `src/detection.py`
| Class / Function | Description |
|---|---|
| `PlayerDetector(is_panoramic, device, det_conf)` | Main detector. Loads YOLOv8m, optionally wraps with SAHI |
| `PlayerDetector.detect(frame)` | Run detection on a single frame → `sv.Detections` |
| `detect_video_mode(video_path)` | Auto-detect FPS, resolution, panoramic mode |

### `src/tracking.py`
| Class / Function | Description |
|---|---|
| `ByteTrackWrapper(fps, activation_threshold)` | Wraps `supervision.ByteTrack` with corrected two-pass config |
| `SORT(max_age, min_hits, iou_thresh)` | IoU-only Kalman tracker (baseline) |
| `SORTWithReID(...)` | SORT + OSNet-AIN appearance features |
| `run_tracking_loop(video_path, detector, tracker, ...)` | Runs full tracking loop → `tracks_df` |
| `run_sort_baseline(tracks_df, frames, ...)` | Runs SORT comparison → `(sort_df, sort_reid_df, reid_available)` |

### `src/ball.py`
| Class / Function | Description |
|---|---|
| `BallKalmanGate(max_jump)` | Kalman filter for ghost-dot rejection |
| `BallDetector(device, use_sahi)` | Ball detection with Kalman gate |
| `run_ball_detection(video_path, frames, ...)` | Full ball detection loop → `ball_df` |

### `src/teams.py`
| Function | Description |
|---|---|
| `classify_teams(tracks_df, video_path)` | Adds `team` column (0 or 1) to tracks_df |

### `src/possession.py`
| Function | Description |
|---|---|
| `compute_possession(tracks_df, ball_df)` | Returns `(poss_df, team_a_pct, team_b_pct)` |

### `src/heatmap.py`
| Function | Description |
|---|---|
| `generate_team_heatmaps(tracks_df, w, h)` | KDE heatmap per team → matplotlib Figure |
| `generate_ball_trajectory(ball_df, ...)` | Ball path plot → matplotlib Figure |
| `generate_possession_chart(poss_df, ...)` | Stacked bar chart → matplotlib Figure |

### `src/video.py`
| Function | Description |
|---|---|
| `render_annotated_video(video_path, ...)` | Render output video with all overlays → file path |

### `src/homography.py`
| Function | Description |
|---|---|
| `get_homography_hybrid(video_path)` | Compute H, H_inv for pitch coordinate mapping |
| `get_pitch_contour_debug(frame)` | Debug view of pitch line detection |

---

## 🔄 Tracker Comparison

| Metric | ByteTrack | SORT | SORT + ReID |
|---|---|---|---|
| Low-conf rescue | ✅ Yes | ❌ No | ❌ No |
| Appearance features | ❌ No | ❌ No | ✅ OSNet-AIN |
| ID switches | **Low** | High | Medium |
| Speed | Fast | Fast | Slow (GPU recommended) |
| Best for | General use | Baseline | Occlusion-heavy scenes |

**Why ByteTrack outperforms SORT:**
ByteTrack keeps a "lost track" buffer and attempts to match low-confidence detections (partially occluded players) before discarding them. SORT discards any detection below its threshold, causing track fragmentation and new IDs when the player reappears.

---

## ⚠️ Known Issues & Notes

### WSL2 / Windows NTFS Slowness
Running on a Windows drive (`/mnt/d/...`) via WSL2 causes very slow Python import times (~2–5 minutes for first boot). To avoid this, copy the project to a Linux-native path:
```bash
cp -r /mnt/d/path/to/MOT-main ~/sports_mot_app
cd ~/sports_mot_app
```

### Re-ID (torchreid) — Optional
`torchreid` is not on PyPI and requires manual installation. If it's missing, SORT+ReID is silently skipped and the app still works fully.

### Ball Model Download
On first run the app downloads `football-ball-detection.pt` (~30MB) from HuggingFace. If this fails (no internet, firewall), it falls back to YOLOv8x COCO "sports ball" class — less accurate but functional.

### Demo Video
`img1.mp4` included in the project is a sample sports clip for quick testing.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.30 | Web dashboard |
| `python-dotenv` | ≥1.0 | `.env` config loading |
| `torch` / `torchvision` | ≥2.0 | Deep learning backend |
| `ultralytics` | ≥8.0 | YOLOv8 detection |
| `supervision` | ≥0.19 | ByteTrack + detection utils |
| `opencv-python-headless` | ≥4.8 | Video I/O, drawing |
| `sahi` | ≥0.11 | Sliced inference for panoramic |
| `filterpy` | ≥1.4 | Kalman filter (ball gate, SORT) |
| `lapjv` | ≥1.3 | Fast Hungarian assignment |
| `scipy` | ≥1.10 | Linear sum assignment |
| `scikit-learn` | ≥1.3 | KMeans team clustering |
| `numpy` | ≥1.24 | Array operations |
| `pandas` | ≥2.0 | Track data tables |
| `matplotlib` | ≥3.7 | Heatmaps, charts |
| `imageio[ffmpeg]` | ≥2.31 | Video encoding |

---

## 👥 Authors

- **Mohamed Ashraf** — E-JUST, CSE 429 Computer Vision Final Project, Spring 2026

---

## 📄 License

This project is for academic purposes — CSE 429 Computer Vision, E-JUST.
