"""
─────────────────────────────────────────────
SportsMOT Local App — Global Configuration
─────────────────────────────────────────────
Central configuration replacing all hardcoded
Kaggle-specific values from the notebook.
─────────────────────────────────────────────
"""

import os
import torch
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ── Paths ─────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.resolve()
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")

def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

def _get_env_float(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"
STATS_OUT_DIR = OUTPUT_DIR / "stats"
HEAT_OUT_DIR  = OUTPUT_DIR / "heatmaps"
MODELS_DIR    = PROJECT_ROOT / "models"

for d in [VIDEO_OUT_DIR, STATS_OUT_DIR, HEAT_OUT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────
_device_env = os.getenv("SPORTSMOT_DEVICE", "").strip()
if _device_env.lower() in ("", "auto"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = _device_env

# ── Detection ─────────────────────────────────
PLAYER_MODEL_NAME = os.getenv("SPORTSMOT_PLAYER_MODEL_NAME", "yolov8m.pt")
PERSON_CLASS_ID   = 0
DET_CONF_PANORAMIC = _get_env_float("SPORTSMOT_DET_CONF_PANORAMIC", 0.15)
DET_CONF_BROADCAST = _get_env_float("SPORTSMOT_DET_CONF_BROADCAST", 0.35)
DET_CONF           = DET_CONF_BROADCAST
DET_IOU           = 0.45          # NMS IoU threshold

# ── SAHI (panoramic slicing) ──────────────────
SAHI_SLICE_H    = 540
SAHI_SLICE_W    = 540
SAHI_OVERLAP    = 0.2
PANORAMIC_RATIO = 3.0             # aspect ratio > this → panoramic mode

# ── ByteTrack ─────────────────────────────────
# FIX #1: Two-pass split uses a higher activation threshold
# so the low-conf rescue pass is meaningful.
BYTETRACK_HIGH_CONF_PANORAMIC = _get_env_float("SPORTSMOT_BYTETRACK_HIGH_CONF_PANORAMIC", 0.35)
BYTETRACK_HIGH_CONF_BROADCAST = _get_env_float("SPORTSMOT_BYTETRACK_HIGH_CONF_BROADCAST", 0.45)
BYTETRACK_ACTIVATION_THRESHOLD = BYTETRACK_HIGH_CONF_BROADCAST
BYTETRACK_LOST_BUFFER          = 30
BYTETRACK_MATCH_THRESHOLD      = 0.8

# ── SORT baseline ─────────────────────────────
SORT_MAX_AGE    = 3
SORT_MIN_HITS   = 2
SORT_IOU_THRESH = 0.25

# ── Ball detection ────────────────────────────
BALL_MODEL_URL  = (
    "https://huggingface.co/SkalskiP/sports.cv/"
    "resolve/main/football-ball-detection.pt"
)
# Fallback URLs to try if primary download fails (401/403)
BALL_MODEL_URL_FALLBACKS = [
    "https://github.com/roboflow/sports/raw/main/data/football-ball-detection.pt",
    "https://media.roboflow.com/football-ball-detection.pt",
]
BALL_MODEL_PATH = MODELS_DIR / "ball_detector.pt"
BALL_CONF       = 0.10
BALL_CONF_DEDICATED = 0.25
BALL_IOU        = 0.45
BALL_SAHI_SLICE = 320
BALL_SAHI_OVERLAP = 0.2
BALL_COCO_CLASS_ID = 32           # fallback: COCO "sports ball"
BALL_DETECT_ON_FULL_RES = True    # always run ball detection on full-res frame

# Kalman gate filter for ball — notebook uses 120 to aggressively reject ghost dots
MAX_BALL_JUMP   = 120             # px — reject ghost detections > this (was 300, notebook=120)

# ── Re-ID (SORT augmentation) ─────────────────
REID_ALPHA = 0.4

# ── Team clustering ───────────────────────────
N_TEAMS         = 2
CROP_TORSO_FRAC = 0.4
MIN_CROP_PX     = 10
N_COLOR_BINS    = 16
KMEANS_SEED     = 42
OVERLAP_IOU_THRESH = 0.3          # IoU above this = overlapping

# ── Possession ────────────────────────────────
MAX_POSSESSION_DIST = 150         # px — max ball-to-player for possession

# ── Heatmap ───────────────────────────────────
HEATMAP_RESOLUTION = (650, 100)
KDE_BANDWIDTH      = 0.12
ALPHA_HEATMAP      = 0.65

# ── Overlays ──────────────────────────────────
DEFENSIVE_TRI_COLOR    = (255, 165, 0)    # orange (BGR)
DEFENSIVE_TRI_ALPHA    = 0.15

# ── Team colors ───────────────────────────────
TEAM_COLORS_BGR = {
    0:  (255, 80,  20),   # Team A → blue (BGR)
    1:  (20,  20,  220),  # Team B → red  (BGR)
    -1: (128, 128, 128),  # unknown → grey
}
TEAM_NAMES = {0: "Team A", 1: "Team B", -1: "Unknown"}

# ── Processing limits ─────────────────────────
MAX_UPLOAD_MB = _get_env_int("SPORTSMOT_MAX_UPLOAD_MB", 800)
MAX_VIDEO_MINUTES = _get_env_int("SPORTSMOT_MAX_VIDEO_MINUTES", 20)
MAX_FRAMES_PANORAMIC = 300
MAX_FRAMES_BROADCAST = None       # no limit

# ── Soccer pitch dimensions (metres) ──────────
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
