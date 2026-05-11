"""
⚽ Sports Player & Ball Tracking Analytics
CSE 429 · Computer Vision · E-JUST

Run: streamlit run app.py
"""
# ── Lightweight imports only (fast) ────────────
import streamlit as st
import cv2
import tempfile
import os
import warnings
# RTX3060
import torch
# OPTIMIZED
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# RTX3060 — 640 keeps both models in 6 GB VRAM simultaneously
INFERENCE_WIDTH = 640
# RTX3060
BALL_DETECT_EVERY = 1  # match notebook: detect every frame for best accuracy
# RTX3060
GPU_VRAM_MB = 6144
MAX_RENDER_DOWNLOAD_MB = 200

# RTX3060
cuda_available = torch.cuda.is_available()
# RTX3060
device = "cuda" if cuda_available else "cpu"
# RTX3060
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    device_name = torch.cuda.get_device_name(0)
    vram_used_mb = torch.cuda.memory_allocated(0) / 1024**2
else:
    device_name = "CPU"
    vram_used_mb = 0.0

# Must be first Streamlit command
st.set_page_config(
    page_title="Sports Tracking Analytics",
    page_icon="⚽",
    layout="wide"
)

# RTX3060
def _maybe_enable_half(model, device):
    # RTX3060
    # Keep model in FP32 and rely on Ultralytics' half flag to avoid dtype issues.
    if device != "cuda":
        return False
    return True

# RTX3060
def _run_yolo(model, half_flag, **kwargs):
    try:
        return model(half=half_flag, **kwargs)[0], half_flag
    except RuntimeError as exc:
        if half_flag and ("Half" in str(exc) or "dtype" in str(exc)):
            model.float()
            return model(half=False, **kwargs)[0], False
        raise

# RTX3060 — cache_resource keeps models resident; half-precision saves ~50% VRAM
@st.cache_resource
def load_models(device):
    """Load player model + ball model (dedicated fine-tuned if available, else COCO fallback).
    Returns (player_model, ball_model, player_half, ball_half, ball_class_id, ball_conf, ball_dedicated).
    """
    import torch, urllib.request
    from ultralytics import YOLO
    from config import (
        BALL_MODEL_URL, BALL_MODEL_URL_FALLBACKS, BALL_MODEL_PATH,
        BALL_CONF, BALL_CONF_DEDICATED, BALL_COCO_CLASS_ID
    )

    player_model = YOLO("yolov8m.pt").to(device)

    # ── Gap 1: try dedicated fine-tuned football-ball-detection.pt ──────────────
    ball_dedicated = False
    if not BALL_MODEL_PATH.exists():
        # Try primary URL, then fallbacks
        urls_to_try = [BALL_MODEL_URL] + BALL_MODEL_URL_FALLBACKS
        for url in urls_to_try:
            try:
                BALL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, str(BALL_MODEL_PATH))
                # Verify the file is valid (not an HTML error page)
                if BALL_MODEL_PATH.stat().st_size > 1_000_000:
                    ball_dedicated = True
                    break
                else:
                    BALL_MODEL_PATH.unlink(missing_ok=True)
            except Exception as _e:
                BALL_MODEL_PATH.unlink(missing_ok=True)
                print(f"Ball model download failed from {url}: ({_e})")
    else:
        ball_dedicated = True

    if ball_dedicated and BALL_MODEL_PATH.exists():
        ball_model    = YOLO(str(BALL_MODEL_PATH)).to(device)
        ball_class_id = 0               # dedicated model: single class 0 = ball
        ball_conf     = BALL_CONF_DEDICATED
    else:
        ball_model    = YOLO("yolov8x.pt").to(device)
        ball_class_id = BALL_COCO_CLASS_ID  # COCO class 32 = sports ball
        ball_conf     = BALL_CONF

    if device == "cuda":
        player_model.model.half()
        ball_model.model.half()
        torch.cuda.empty_cache()

    player_half = _maybe_enable_half(player_model, device)
    ball_half   = _maybe_enable_half(ball_model, device)
    return player_model, ball_model, player_half, ball_half, ball_class_id, ball_conf, ball_dedicated


@st.cache_resource
def load_sahi_player(device):
    """Gap 2: SAHI AutoDetectionModel for panoramic player detection.
    Returns the sahi model or None if sahi is unavailable.
    """
    try:
        from sahi import AutoDetectionModel
        from config import DET_CONF_PANORAMIC, SAHI_SLICE_W, SAHI_SLICE_H, SAHI_OVERLAP
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolov8m.pt",
            confidence_threshold=DET_CONF_PANORAMIC,
            device=device,
        )
        return sahi_model
    except Exception as _e:
        print(f"SAHI not available for panoramic detection: {_e}")
        return None


@st.cache_resource
def load_sahi_ball(device, ball_model_path_str, ball_conf):
    """Gap 1+2: SAHI wrapper for ball detection (helps find small balls in panoramic).
    Returns sahi model or None.
    """
    try:
        from sahi import AutoDetectionModel
        sahi_ball = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=ball_model_path_str,
            confidence_threshold=ball_conf,
            device=device,
        )
        return sahi_ball
    except Exception as _e:
        print(f"SAHI for ball not available: {_e}")
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Session State Initialisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RTX3060
_STATE_KEYS = [
    "tracks_df", "ball_df", "poss_df", "sort_df", "sort_reid_df",
    "frames_run", "fps", "img_w", "img_h", "is_panoramic",
    "vname", "a_pct", "b_pct", "reid_available",
    "rendered_video_path", "rendered_video_name",
    "analysis_done", "video_path",
    "bt_high_conf",
    "gpu_peak_mb", "inference_seconds", "avg_fps", "device_name",
]
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None
for _legacy in ("vid_bytes", "vid_filename"):
    if _legacy in st.session_state:
        st.session_state.pop(_legacy, None)
if not st.session_state.get("analysis_done"):
    st.session_state["analysis_done"] = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Header  (renders instantly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("⚽ Sports Player & Ball Tracking")
st.caption("CSE 429 · Computer Vision and Pattern Recognition · E-JUST")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sidebar Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.sidebar.header("⚙️ Configuration")
# RTX3060
st.sidebar.markdown(f"**CUDA available:** {cuda_available}")
# RTX3060
if not cuda_available:
    st.sidebar.warning("CUDA not available — install CUDA-enabled PyTorch (cu124) to use the GPU.")
# RTX3060
if device == "cuda":
    st.sidebar.markdown(f"**Device:** {device_name}")
    st.sidebar.markdown(f"**VRAM in use:** {vram_used_mb:.0f} MB")
# RTX3060
max_frames     = st.sidebar.slider("Max frames to process", 30, 750, 200, 10)
# RTX3060
if device == "cuda":
    st.sidebar.caption("RTX 3060 detected — 200 frames safe default (≈1 min)")
st.sidebar.markdown("---")
st.sidebar.subheader("Features")
enable_ball    = st.sidebar.checkbox("🏐 Ball detection", value=True)
# OPTIMIZED
ball_detect_every = st.sidebar.slider(
    "Ball detect interval (frames)", 1, 5, BALL_DETECT_EVERY, 1,
    help="Run ball detection every N frames; skipped frames use Kalman prediction."
)
enable_sort    = st.sidebar.checkbox("📊 SORT comparison", value=True)
enable_triangle= st.sidebar.checkbox("🔺 Defensive triangle", value=False,
                                     help="Requires homography calibration")
attacking_team = st.sidebar.selectbox("Attacking team", [0, 1],
                                      format_func=lambda x: f"Team {'A' if x==0 else 'B'}")
# OPTIMIZED
st.sidebar.markdown("[GitHub repo](https://github.com/MohAsh-arch/sports-mot-bytetrack)")

# ── Gap 4: Ground-truth file for real MOTA / IDF1 metrics ─────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Ground Truth (optional)")
st.sidebar.caption("Upload MOT-format GT file to compute MOTA/IDF1 in the Comparison tab.")
_gt_uploaded = st.sidebar.file_uploader(
    "GT annotation file (.txt)", type=["txt", "csv"],
    help="MOT format: frame,id,x,y,w,h,conf,cls,vis",
    key="gt_file_uploader",
)
_gt_path = None
if _gt_uploaded is not None:
    _gt_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    _gt_tmp.write(_gt_uploaded.read()); _gt_tmp.close()
    _gt_path = _gt_tmp.name
    st.sidebar.success(f"GT loaded: {_gt_uploaded.name}")

# ── New Analysis button ────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("🆕 New Analysis", use_container_width=True,
                     help="Clear all results and start fresh with a new video"):
    # Wipe every tracked key so the app returns to its initial state
    for _k in _STATE_KEYS:
        st.session_state[_k] = None
    st.session_state["analysis_done"] = False
    # Clean up any leftover temp video file
    _old_path = st.session_state.get("video_path")
    if _old_path and os.path.exists(str(_old_path)):
        try:
            os.unlink(_old_path)
        except OSError:
            pass
    st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  File Upload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
uploaded = st.file_uploader("📁 Upload a sports video", type=["mp4", "avi", "mov"])
use_demo = st.checkbox("Or use bundled demo video (img1.mp4)")

# --- Lazy config load (only when we have a file) ---
MAX_UPLOAD_MB    = 800
MAX_VIDEO_MINUTES= 20

video_path = None
if uploaded:
    if hasattr(uploaded, "size"):
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"Upload too large ({size_mb:.1f} MB). Limit is {MAX_UPLOAD_MB} MB.")
            st.stop()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(uploaded.read()); tmp.close()
    video_path = tmp.name
elif use_demo:
    demo = Path(__file__).parent / "img1.mp4"
    if demo.exists():
        video_path = str(demo)
    else:
        st.error("Demo video img1.mp4 not found in project root.")

if video_path is None:
    st.info("👆 Upload a video or select the demo to start analysis.")
    if st.session_state.get("analysis_done"):
        st.info("ℹ️ Previous analysis results are still available below — scroll down.")
    if not st.session_state.get("analysis_done"):
        st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Quick video probe (cv2 only — no torch needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if video_path:
    cap_probe = cv2.VideoCapture(str(video_path))
    if not cap_probe.isOpened():
        st.error("Could not open the uploaded video. Try a different file.")
        if uploaded and os.path.exists(video_path):
            os.unlink(video_path)
        st.stop()

    probe_fps    = cap_probe.get(cv2.CAP_PROP_FPS) or 0
    probe_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    probe_w      = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    probe_h      = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    if probe_fps <= 0 or probe_frames <= 0:
        st.error("Uploaded video appears to be corrupted or unsupported.")
        if uploaded and os.path.exists(video_path):
            os.unlink(video_path)
        st.stop()

    video_minutes = probe_frames / max(probe_fps, 1) / 60
    if video_minutes > MAX_VIDEO_MINUTES:
        st.warning(
            f"Long video detected (~{video_minutes:.1f} min). "
            f"Processing will be capped to {MAX_VIDEO_MINUTES} minutes."
        )

    is_panoramic_probe = (probe_w / max(probe_h, 1)) > 3.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resolution", f"{probe_w}×{probe_h}")
    col2.metric("FPS", round(probe_fps, 1))
    col3.metric("Total Frames", probe_frames)
    col4.metric("Mode", "Panoramic" if is_panoramic_probe else "Broadcast")

    st.markdown("---")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Run Analysis Button
    #  Heavy imports happen ONLY here, after click
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        for _k in _STATE_KEYS:
            st.session_state[_k] = None
        st.session_state["analysis_done"] = False

        with st.spinner("⏳ Loading ML models (first run takes ~2 min on WSL2)…"):
            # Heavy imports — only executed when user clicks Run
            # OPTIMIZED
            import supervision as sv
            # RTX3060
            from config import (
                OUTPUT_DIR, VIDEO_OUT_DIR, STATS_OUT_DIR, HEAT_OUT_DIR,
                MAX_FRAMES_PANORAMIC, MAX_FRAMES_BROADCAST,
                DET_CONF_PANORAMIC, DET_CONF_BROADCAST,
                BYTETRACK_HIGH_CONF_PANORAMIC, BYTETRACK_HIGH_CONF_BROADCAST,
                REID_ALPHA, MAX_UPLOAD_MB as _MAX_MB, MAX_VIDEO_MINUTES as _MAX_MIN,
                DET_IOU,
            )
            from src.detection import detect_video_mode
            from src.tracking import ByteTrackWrapper, run_sort_baseline
            from src.ball import BallKalmanGate
            from src.teams import classify_teams
            from src.possession import compute_possession
            from src.heatmap import generate_team_heatmaps, generate_ball_trajectory, generate_possession_chart
            from src.video import render_annotated_video
            from src.evaluation import compare_trackers
            from src.homography import get_homography_hybrid, get_pitch_contour_debug

            # RTX3060 + Gap 1: dedicated ball model if available
            (
                player_model, ball_model, player_half, ball_half,
                ball_class_id, ball_conf, ball_dedicated
            ) = load_models(device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(0)

        fps, img_w, img_h, total_frames, is_panoramic = detect_video_mode(video_path)
        limit_by_minutes = int(MAX_VIDEO_MINUTES * 60 * fps)
        frames_to_run = min(max_frames, total_frames, limit_by_minutes)
        bt_high_conf  = BYTETRACK_HIGH_CONF_PANORAMIC if is_panoramic else BYTETRACK_HIGH_CONF_BROADCAST
        vname         = Path(video_path).stem

        # ── Gap 2: load SAHI models ─────────────────────────
        _sahi_player = None
        _sahi_ball   = None
        with st.spinner("🔭 Loading SAHI models (always used for ball detection to improve accuracy)..."):
            from config import SAHI_SLICE_H, SAHI_SLICE_W, SAHI_OVERLAP, BALL_SAHI_SLICE, BALL_SAHI_OVERLAP
            _ball_path   = str(BALL_MODEL_PATH) if ball_dedicated else "yolov8x.pt"
            _sahi_ball   = load_sahi_ball(device, _ball_path, ball_conf)
            if is_panoramic:
                _sahi_player = load_sahi_player(device)
                st.sidebar.info(f"🔭 Panoramic mode: Player SAHI {'✅' if _sahi_player else '❌'}")

        # ── Step 1: Detection + Tracking ──────────
        _ball_label = f"dedicated ({ball_conf:.2f})" if ball_dedicated else f"COCO class 32 ({ball_conf:.2f})"
        with st.status("🔍 Step 1/5: Detection & Tracking...", expanded=True) as status:
            progress = st.progress(0)
            st.write(f"Processing {frames_to_run} frames | ByteTrack thresh={bt_high_conf} | "
                     f"Ball model: {_ball_label} | Panoramic={is_panoramic}")

            def track_progress(idx, total, elapsed):
                pct = idx / total
                eta = (total - idx) / (idx / elapsed) if idx > 0 else 0
                progress.progress(pct, f"Frame {idx}/{total} | ETA: {eta:.0f}s")

            # OPTIMIZED
            def detect_pitch_bounds(frame):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                green_lower = np.array([25, 30, 40])
                green_upper = np.array([95, 255, 255])
                green_mask = cv2.inRange(hsv, green_lower, green_upper)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
                green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return 0, frame.shape[0], False

                largest = max(contours, key=cv2.contourArea)
                _, y, _, h = cv2.boundingRect(largest)
                y_top = max(0, y)
                y_bot = min(frame.shape[0], y + h)
                if y_bot - y_top < 10:
                    return 0, frame.shape[0], False
                return y_top, y_bot, True

            # OPTIMIZED
            tracker = ByteTrackWrapper(fps=fps, activation_threshold=bt_high_conf)
            # OPTIMIZED
            ball_gate = BallKalmanGate() if enable_ball else None
            # OPTIMIZED
            cap = cv2.VideoCapture(str(video_path))
            tracker.reset(); rows = []; ball_rows = []; idx = 0; t0 = time.time()

            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                st.error("Could not read video frames.")
                st.stop()

            y_top, y_bot, pitch_ok = detect_pitch_bounds(first_frame)
            if not pitch_ok:
                y_top, y_bot = 0, img_h
            crop_h = max(1, y_bot - y_top)
            scale = INFERENCE_WIDTH / max(img_w, 1)
            inf_h = max(1, int(round(crop_h * scale)))

            res_factor = (img_w * img_h) / max(1, (INFERENCE_WIDTH * inf_h))
            ball_factor = float(ball_detect_every) if enable_ball else 1.0
            roi_factor = (img_h / crop_h) if pitch_ok else 1.0
            speedup = res_factor * ball_factor * roi_factor

            _mode_label = "Panoramic+SAHI" if (is_panoramic and _sahi_player) else ("Panoramic (no SAHI)" if is_panoramic else "Broadcast")
            st.sidebar.info(
                f"**Device:** {device}\n\n"
                f"**Mode:** {_mode_label}\n\n"
                f"**Inference resolution:** {INFERENCE_WIDTH}x{inf_h}\n\n"
                f"**Ball detect interval:** {ball_detect_every}\n\n"
                f"**Estimated speedup:** {speedup:.1f}x"
            )

            # RTX3060
            with torch.inference_mode():
                while idx < frames_to_run:
                    if idx == 0:
                        frame = first_frame
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            break

                    if pitch_ok:
                        crop = frame[y_top:y_bot, :]
                        if crop.size == 0:
                            crop = frame
                    else:
                        crop = frame

                    # ── Gap 2: SAHI slicing for panoramic, direct resize for broadcast ─
                    if is_panoramic and _sahi_player is not None:
                        # SAHI slices the crop into 540×540 tiles — finds small/far players
                        from sahi.predict import get_sliced_prediction
                        _sahi_result = get_sliced_prediction(
                            crop,
                            _sahi_player,
                            slice_height=SAHI_SLICE_H,
                            slice_width=SAHI_SLICE_W,
                            overlap_height_ratio=SAHI_OVERLAP,
                            overlap_width_ratio=SAHI_OVERLAP,
                            verbose=0,
                        )
                        _boxes, _confs, _clss = [], [], []
                        for _obj in _sahi_result.object_prediction_list:
                            if _obj.category.name.lower() != "person":
                                continue
                            if _obj.score.value < DET_CONF_PANORAMIC:
                                continue
                            _b = _obj.bbox
                            # crop coords → add y_top to get full-frame y
                            _y1c = _b.miny + (y_top if pitch_ok else 0)
                            _y2c = _b.maxy + (y_top if pitch_ok else 0)
                            _boxes.append([_b.minx, _y1c, _b.maxx, _y2c])
                            _confs.append(_obj.score.value)
                            _clss.append(0)
                        if _boxes:
                            _ba = np.array(_boxes, dtype=np.float32)
                            _ba[:, [0, 2]] = np.clip(_ba[:, [0, 2]], 0, img_w - 1)
                            _ba[:, [1, 3]] = np.clip(_ba[:, [1, 3]], 0, img_h - 1)
                            player_dets = sv.Detections(
                                xyxy=_ba,
                                confidence=np.array(_confs, dtype=np.float32),
                                class_id=np.array(_clss, dtype=int),
                            )
                        else:
                            player_dets = sv.Detections.empty()
                        infer_frame = crop  # keep ref for ball detection below
                    else:
                        # Broadcast mode: resize + direct YOLO (GPU-optimised)
                        infer_frame = cv2.resize(
                            crop, (INFERENCE_WIDTH, inf_h), interpolation=cv2.INTER_LINEAR
                        )
                        _imgsz = max(32, (INFERENCE_WIDTH // 32) * 32)
                        player_results, player_half = _run_yolo(
                            player_model,
                            player_half,
                            source=infer_frame,
                            conf=DET_CONF_BROADCAST,
                            iou=DET_IOU,
                            classes=[0],
                            imgsz=_imgsz,
                            batch=1,
                            device=device,
                            verbose=False,
                        )
                        player_boxes_obj = player_results.boxes
                        if player_boxes_obj is not None and len(player_boxes_obj) > 0:
                            boxes = player_boxes_obj.xyxy.cpu().numpy()
                            confs = player_boxes_obj.conf.cpu().numpy()
                            clss  = player_boxes_obj.cls.cpu().numpy().astype(int)
                            boxes[:, [0, 2]] /= scale
                            boxes[:, [1, 3]] /= scale
                            if pitch_ok:
                                boxes[:, [1, 3]] += y_top
                            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w - 1)
                            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h - 1)
                            player_mask = confs >= 0.35
                            if player_mask.any():
                                player_dets = sv.Detections(
                                    xyxy=boxes[player_mask].astype(np.float32),
                                    confidence=confs[player_mask].astype(np.float32),
                                    class_id=clss[player_mask].astype(int),
                                )
                            else:
                                player_dets = sv.Detections.empty()
                        else:
                            player_dets = sv.Detections.empty()

                    # ── Gap 1: ball detection using dedicated model + correct class/conf ─
                    ball_candidate = None
                    if enable_ball and idx % ball_detect_every == 0:
                        # ── CRITICAL FIX: detect ball on FULL-RESOLUTION frame ──
                        # The notebook (Cell 5) runs SAHI on the full frame because
                        # the ball is tiny (8-15px panoramic, 20-40px broadcast).
                        # Downscaling to 640px makes it 1-3px → undetectable.
                        ball_detect_frame = frame  # full resolution, not crop/infer_frame
                        if _sahi_ball is not None:
                            # Use SAHI slicing for small ball in ALL footage (panoramic and broadcast)
                            from sahi.predict import get_sliced_prediction
                            _br = get_sliced_prediction(
                                ball_detect_frame,
                                _sahi_ball,
                                slice_height=BALL_SAHI_SLICE,
                                slice_width=BALL_SAHI_SLICE,
                                overlap_height_ratio=BALL_SAHI_OVERLAP,
                                overlap_width_ratio=BALL_SAHI_OVERLAP,
                                verbose=0,
                            )
                            _bpreds = [
                                o for o in _br.object_prediction_list
                                if o.score.value >= ball_conf
                                and (ball_dedicated or o.category.id == ball_class_id)
                            ]
                            if _bpreds:
                                _best = max(_bpreds, key=lambda o: o.score.value)
                                _bb = _best.bbox
                                # No y_top offset needed — we're using full frame
                                bc_x = float(np.clip((_bb.minx+_bb.maxx)/2, 0, img_w-1))
                                bc_y = float(np.clip((_bb.miny+_bb.maxy)/2, 0, img_h-1))
                                ball_candidate = (bc_x, bc_y, float(_best.score.value))
                        else:
                            # Direct inference on full-resolution frame (not downscaled)
                            _ball_classes = None if ball_dedicated else [ball_class_id]
                            ball_results, ball_half = _run_yolo(
                                ball_model,
                                ball_half,
                                source=ball_detect_frame,
                                conf=ball_conf,
                                iou=DET_IOU,
                                classes=_ball_classes,
                                imgsz=640,
                                batch=1,
                                device=device,
                                verbose=False,
                            )
                            ball_boxes_obj = ball_results.boxes
                            if ball_boxes_obj is not None and len(ball_boxes_obj) > 0:
                                ball_boxes = ball_boxes_obj.xyxy.cpu().numpy()
                                ball_confs = ball_boxes_obj.conf.cpu().numpy()
                                # filter to ball class if not dedicated
                                if not ball_dedicated:
                                    _bc_mask = ball_boxes_obj.cls.cpu().numpy().astype(int) == ball_class_id
                                    if _bc_mask.any():
                                        ball_boxes = ball_boxes[_bc_mask]
                                        ball_confs = ball_confs[_bc_mask]
                                    else:
                                        ball_boxes = np.empty((0, 4))
                                if len(ball_boxes) > 0:
                                    # No rescaling needed — inference was on full frame
                                    ball_boxes[:, [0, 2]] = np.clip(ball_boxes[:, [0, 2]], 0, img_w-1)
                                    ball_boxes[:, [1, 3]] = np.clip(ball_boxes[:, [1, 3]], 0, img_h-1)
                                    best_idx = int(np.argmax(ball_confs))
                                    bx = ball_boxes[best_idx]
                                    bc_x = float(np.clip((bx[0]+bx[2])/2, 0, img_w-1))
                                    bc_y = float(np.clip((bx[1]+bx[3])/2, 0, img_h-1))
                                    ball_candidate = (bc_x, bc_y, float(ball_confs[best_idx]))

                    tracked = tracker.update(player_dets)
                    if tracked is not None and len(tracked) > 0:
                        for i in range(len(tracked)):
                            x1, y1, x2, y2 = tracked.xyxy[i]
                            rows.append({"frame": idx, "track_id": int(tracked.tracker_id[i]),
                                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                                "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                                "cx": round(float((x1 + x2) / 2), 1), "cy": round(float((y1 + y2) / 2), 1),
                                "conf": round(float(tracked.confidence[i]), 3),
                                "width": round(float(x2 - x1), 1), "height": round(float(y2 - y1), 1)})

                    if enable_ball:
                        ball_row = {
                            "frame": idx,
                            "cx": float("nan"),
                            "cy": float("nan"),
                            "conf": float("nan"),
                            "is_interpolated": False,
                        }
                        if idx % ball_detect_every == 0:
                            if ball_candidate is not None:
                                cx, cy, conf = ball_candidate
                                if ball_gate.update(cx, cy):
                                    ball_row["cx"] = round(cx, 1)
                                    ball_row["cy"] = round(cy, 1)
                                    ball_row["conf"] = round(conf, 3)
                        else:
                            # Skipped frame: use non-mutating prediction
                            # (get_predicted now returns F@x without calling kf.predict,
                            #  avoiding the double-predict drift bug)
                            pred = ball_gate.get_predicted()
                            if pred is not None:
                                px = float(np.clip(pred[0], 0, img_w - 1))
                                py = float(np.clip(pred[1], 0, img_h - 1))
                                ball_row["cx"] = round(px, 1)
                                ball_row["cy"] = round(py, 1)
                                ball_row["conf"] = 0.0
                                ball_row["is_interpolated"] = True
                        ball_rows.append(ball_row)

                    if idx % 5 == 0:
                        track_progress(idx, frames_to_run, time.time() - t0)
                        # Proactively free GPU memory every 5 frames
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    idx += 1

            cap.release()
            # free last frame tensors (player_results only exists in broadcast path)
            try:
                del infer_frame, player_results
            except NameError:
                pass
            # RTX3060
            inference_seconds = max(time.time() - t0, 1e-6)
            # RTX3060
            avg_fps = (idx / inference_seconds) if inference_seconds > 0 else 0.0
            # RTX3060
            gpu_peak_mb = 0.0
            # RTX3060
            if device == "cuda":
                gpu_peak_mb = torch.cuda.max_memory_allocated(0) / 1024**2
                torch.cuda.empty_cache()
            tracks_df = pd.DataFrame(rows)
            if tracks_df.empty:
                tracks_df = pd.DataFrame(columns=[
                    "frame", "track_id", "x1", "y1", "x2", "y2",
                    "cx", "cy", "conf", "width", "height",
                ])
            frames_run = idx
            if enable_ball:
                ball_df = pd.DataFrame(ball_rows)
                if ball_df.empty:
                    ball_df = pd.DataFrame(columns=["frame", "cx", "cy", "conf", "is_interpolated"])
            else:
                ball_df = None

            progress.progress(1.0, "✅ Tracking complete")
            st.write(f"**{tracks_df['track_id'].nunique()} unique player IDs** across {frames_run} frames")
            status.update(label="✅ Step 1: Detection & Tracking complete", state="complete")

        # ── Step 2: Ball Detection ────────────────
        # OPTIMIZED
        if enable_ball:
            with st.status("🏐 Step 2/5: Ball Detection...", expanded=True) as status:
                # OPTIMIZED
                progress2 = st.progress(1.0)
                st.write("Single-pass detector + Kalman interpolation (from Step 1).")

                if ball_df is None or ball_df.empty:
                    st.warning("No ball detections found.")
                else:
                    interp_mask = ball_df["is_interpolated"] if "is_interpolated" in ball_df.columns else False
                    detected_mask = ball_df["conf"].notna() & ~interp_mask
                    detected = int(detected_mask.sum())
                    pct = (detected / frames_run * 100) if frames_run > 0 else 0
                    progress2.progress(1.0, "✅ Ball detection complete")
                    st.write(f"**Ball detected in {detected}/{frames_run} frames** ({pct:.1f}%)")
                status.update(label="✅ Step 2: Ball Detection complete", state="complete")
        else:
            st.info("Ball detection skipped.")

        # ── Step 3: Team Classification ───────────
        with st.status("👕 Step 3/5: Team Classification...", expanded=True) as status:
            st.write("Overlap-aware clustering (RGB+HSV features)...")
            tracks_df, team_map = classify_teams(tracks_df, video_path)
            ta = (tracks_df["team"] == 0).sum()
            tb = (tracks_df["team"] == 1).sum()
            st.write(f"**Team A:** {ta} detections | **Team B:** {tb} detections")
            status.update(label="✅ Step 3: Team Classification complete", state="complete")

        # ── Step 4: Possession ────────────────────
        poss_df = None; a_pct = b_pct = 0
        if ball_df is not None:
            with st.status("📊 Step 4/5: Possession Analysis...", expanded=True) as status:
                poss_df, a_pct, b_pct = compute_possession(tracks_df, ball_df)
                st.write(f"**Team A:** {a_pct:.1f}% | **Team B:** {b_pct:.1f}%")
                status.update(label="✅ Step 4: Possession Analysis complete", state="complete")

        # ── Step 5: SORT Comparison ───────────────
        sort_df = None; sort_reid_df = None; reid_available = False
        if enable_sort:
            with st.status("🔄 Step 5/5: SORT Comparison...", expanded=True) as status:
                # OPTIMIZED
                sort_df, sort_reid_df, reid_available = run_sort_baseline(
                    tracks_df, frames_run, video_path=video_path, device=device)
                n_bt   = tracks_df['track_id'].nunique()
                n_sort = sort_df['track_id'].nunique()
                label  = f"ByteTrack: {n_bt} IDs | SORT: {n_sort} IDs"
                if sort_reid_df is not None:
                    label += f" | SORT+ReID: {sort_reid_df['track_id'].nunique()} IDs"
                st.write(label)
                status.update(label="✅ Step 5: SORT Comparison complete", state="complete")

        # ── Save CSVs ─────────────────────────────
        tracks_df.to_csv(STATS_OUT_DIR / f"{vname}_tracks.csv", index=False)
        if ball_df   is not None: ball_df.to_csv(STATS_OUT_DIR / f"{vname}_ball.csv", index=False)
        if poss_df   is not None: poss_df.to_csv(STATS_OUT_DIR / f"{vname}_possession.csv", index=False)
        if sort_df   is not None: sort_df.to_csv(STATS_OUT_DIR / f"{vname}_sort.csv", index=False)
        if sort_reid_df is not None: sort_reid_df.to_csv(STATS_OUT_DIR / f"{vname}_sort_reid.csv", index=False)

        # ── Persist to session_state ──────────────
        # RTX3060
        st.session_state.update({
            "tracks_df":     tracks_df,
            "ball_df":       ball_df,
            "poss_df":       poss_df,
            "sort_df":       sort_df,
            "sort_reid_df":  sort_reid_df,
            "frames_run":    frames_run,
            "fps":           fps,
            "img_w":         img_w,
            "img_h":         img_h,
            "is_panoramic":  is_panoramic,
            "vname":         vname,
            "a_pct":         a_pct,
            "b_pct":         b_pct,
            "reid_available":reid_available,
            "video_path":    video_path,
            "bt_high_conf":  bt_high_conf,
            "gpu_peak_mb":   gpu_peak_mb,
            "inference_seconds": inference_seconds,
            "avg_fps":       avg_fps,
            "device_name":   device_name,
            "analysis_done": True,
        })

        if uploaded and video_path and os.path.exists(video_path):
            os.unlink(video_path)

        st.success("✅ All analysis complete!")
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Results — outside button block so widget
#  interactions never wipe the results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.get("analysis_done"):
    tracks_df     = st.session_state["tracks_df"]
    ball_df       = st.session_state["ball_df"]
    poss_df       = st.session_state["poss_df"]
    sort_df       = st.session_state["sort_df"]
    sort_reid_df  = st.session_state["sort_reid_df"]
    frames_run    = st.session_state["frames_run"]
    _fps          = st.session_state["fps"]
    _img_w        = st.session_state["img_w"]
    _img_h        = st.session_state["img_h"]
    vname         = st.session_state["vname"]
    a_pct         = st.session_state["a_pct"]
    b_pct         = st.session_state["b_pct"]
    reid_available= st.session_state["reid_available"]
    _video_path   = st.session_state.get("video_path", "")
    _bt_high_conf = st.session_state.get("bt_high_conf", 0.45)
    # RTX3060
    gpu_peak_mb = st.session_state.get("gpu_peak_mb")
    # RTX3060
    inference_seconds = st.session_state.get("inference_seconds")
    # RTX3060
    avg_fps = st.session_state.get("avg_fps")
    # RTX3060
    device_name_state = st.session_state.get("device_name", device_name)

    if video_path is None:
        st.markdown("---")

    # RTX3060
    if device == "cuda" and gpu_peak_mb is not None and inference_seconds is not None and avg_fps is not None:
        device_label = "RTX 3060" if "RTX 3060" in device_name_state else device_name_state
        st.sidebar.info(
            f"**Device:** {device_label}\n\n"
            f"**Peak VRAM used:** {gpu_peak_mb:.0f} MB / {GPU_VRAM_MB} MB\n\n"
            f"**Total inference time:** {inference_seconds:.1f} seconds\n\n"
            f"**Average FPS:** {avg_fps:.1f} frames/sec"
        )

    st.success("✅ Analysis results ready.")
    st.markdown("---")

    tabs = st.tabs(["📹 Preview", "📊 Possession", "🔥 Heatmaps",
                    "⚽ Ball Trajectory", "🔄 ByteTrack vs SORT"])

    # ── Tab 1: Preview ────────────────────────
    with tabs[0]:
        st.subheader("Annotated Frame Preview")
        frame_num = st.slider("Select frame", 0, frames_run - 1, frames_run // 2, key="frame_slider")

        if _video_path and os.path.exists(str(_video_path)):
            cap = cv2.VideoCapture(str(_video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, preview_frame = cap.read()
            cap.release()
            if ret:
                ann = preview_frame.copy()
                ft  = tracks_df[tracks_df["frame"] == frame_num]
                for _, row in ft.iterrows():
                    tid   = int(row["track_id"]); team = int(row["team"])
                    color = {0:(255,80,20), 1:(20,20,220), -1:(128,128,128)}.get(team,(128,128,128))
                    x1,y1,x2,y2 = int(row["x1"]),int(row["y1"]),int(row["x2"]),int(row["y2"])
                    cv2.rectangle(ann, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(ann, f"#{tid}", (x1,max(12,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if ball_df is not None:
                    br = ball_df[ball_df["frame"] == frame_num]
                    # OPTIMIZED
                    if not br.empty and not np.isnan(br.iloc[0]["cx"]):
                        bx, by = int(br.iloc[0]["cx"]), int(br.iloc[0]["cy"])
                        is_interp = bool(br.iloc[0]["is_interpolated"]) if "is_interpolated" in br.columns else False
                        color = (160, 160, 160) if is_interp else (0, 255, 0)
                        cv2.circle(ann, (bx, by), 14, color, 3)
                st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                         caption=f"Frame {frame_num}", use_container_width=True)

                show_debug = st.checkbox("Show pitch contour debug", value=False, key="debug_cb")
                if show_debug:
                    from src.homography import get_pitch_contour_debug
                    dbg, ok = get_pitch_contour_debug(preview_frame)
                    if dbg is not None:
                        st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
                                 caption="Pitch contour + corner detection",
                                 use_container_width=True)
                    if not ok:
                        st.info("Pitch contour not detected in this frame.")
        else:
            st.info("Original video file not available for frame preview.")

        st.markdown("---")
        st.subheader("🎬 Render Annotated Video")

        rendered_path = st.session_state.get("rendered_video_path")
        rendered_name = st.session_state.get("rendered_video_name")
        if rendered_path and os.path.exists(str(rendered_path)):
            if not rendered_name:
                rendered_name = Path(rendered_path).name
            size_mb = os.path.getsize(rendered_path) / (1024 * 1024)
            st.success(f"Video ready: {rendered_name} ({size_mb:.1f} MB)")
            if size_mb <= MAX_RENDER_DOWNLOAD_MB:
                with open(rendered_path, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download rendered video",
                        data=vf,
                        file_name=rendered_name,
                        mime="video/mp4",
                        key="dl_btn_rendered",
                    )
            else:
                st.warning(
                    f"Rendered video is {size_mb:.1f} MB. "
                    "Use the file path below to copy it manually."
                )
                st.code(str(rendered_path))
            if st.button("🔄 Re-render video", key="rerender_btn"):
                st.session_state["rendered_video_path"] = None
                st.session_state["rendered_video_name"] = None
                st.rerun()
        else:
            if _video_path and os.path.exists(str(_video_path)):
                if st.button("🎬 Render full annotated video", key="render_btn"):
                    with st.spinner("Rendering video… this may take a minute"):
                        from src.homography import get_homography_hybrid
                        from src.video import render_annotated_video
                        H, H_inv  = get_homography_hybrid(_video_path)
                        # OPTIMIZED
                        vid_path  = render_annotated_video(
                            _video_path, tracks_df, (ball_df if ball_df is not None else pd.DataFrame()),
                            poss_df, frames_run, _fps, vname,
                            H, H_inv, attacking_team, enable_triangle)
                        st.session_state["rendered_video_path"] = str(vid_path)
                        st.session_state["rendered_video_name"] = f"{vname}_final.mp4"
                    st.rerun()
            else:
                st.info("Original video not available for rendering.")

    # ── Tab 2: Possession ─────────────────────
    with tabs[1]:
        if poss_df is not None:
            st.subheader("Ball Possession")
            pc1, pc2 = st.columns(2)
            pc1.metric("Team A", f"{a_pct:.1f}%")
            pc2.metric("Team B", f"{b_pct:.1f}%")
            from src.heatmap import generate_possession_chart
            fig_poss = generate_possession_chart(poss_df, a_pct, b_pct, _fps)
            st.pyplot(fig_poss)
        else:
            st.info("Enable ball detection for possession analysis.")

    # ── Tab 3: Heatmaps ───────────────────────
    with tabs[2]:
        st.subheader("Player Position Heatmaps")
        if tracks_df.empty or "team" not in tracks_df.columns:
            st.info("Not enough tracking data to build heatmaps.")
        else:
            from src.heatmap import generate_team_heatmaps
            fig_heat = generate_team_heatmaps(tracks_df, _img_w, _img_h)
            st.pyplot(fig_heat)

    # ── Tab 4: Ball Trajectory ────────────────
    with tabs[3]:
        if ball_df is not None:
            st.subheader("Ball Trajectory (Kalman-filtered)")
            from src.heatmap import generate_ball_trajectory
            fig_traj = generate_ball_trajectory(ball_df, frames_run, _fps, _img_w, _img_h)
            st.pyplot(fig_traj)
        else:
            st.info("Enable ball detection for trajectory.")

    # ── Tab 5: Comparison ─────────────────────
    with tabs[4]:
        st.subheader("ByteTrack vs SORT Comparison")
        if sort_df is not None:
            st.markdown(f"""
            **Why ByteTrack outperforms SORT:**
            - ByteTrack uses a **two-pass** matching strategy
            - High-confidence detections (≥{_bt_high_conf}) match first
            - Low-confidence detections get a **rescue pass**
            - SORT discards low-confidence detections → more ID switches
            """)
            from src.evaluation import compare_trackers
            # Gap 4: pass GT file if uploaded — enables real MOTA/IDF1 metrics
            _gt_file_for_eval = _gt_path if _gt_path and os.path.exists(_gt_path) else ""
            if _gt_file_for_eval:
                st.info(f"📋 Computing MOTA/IDF1 with uploaded GT file")
            else:
                st.caption("Upload a GT .txt file in the sidebar to see MOTA/IDF1 metrics.")
            comp_df, fig_comp = compare_trackers(tracks_df, sort_df, sort_reid_df, _gt_file_for_eval, frames_run)
            st.dataframe(comp_df, use_container_width=True)
            st.pyplot(fig_comp)
            if not reid_available:
                st.info("Re-ID not available (torchreid not installed). SORT+ReID skipped.")
        else:
            st.info("Enable SORT comparison in sidebar.")
