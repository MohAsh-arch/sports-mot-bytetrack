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
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Must be first Streamlit command
st.set_page_config(
    page_title="Sports Tracking Analytics",
    page_icon="⚽",
    layout="wide"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Session State Initialisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_STATE_KEYS = [
    "tracks_df", "ball_df", "poss_df", "sort_df", "sort_reid_df",
    "frames_run", "fps", "img_w", "img_h", "is_panoramic",
    "vname", "a_pct", "b_pct", "reid_available",
    "vid_bytes", "vid_filename",
    "analysis_done", "video_path",
    "bt_high_conf",
]
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None
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
max_frames     = st.sidebar.slider("Max frames to process", 30, 500, 100, 10)
st.sidebar.markdown("---")
st.sidebar.subheader("Features")
enable_ball    = st.sidebar.checkbox("🏐 Ball detection", value=True)
enable_sort    = st.sidebar.checkbox("📊 SORT comparison", value=True)
enable_triangle= st.sidebar.checkbox("🔺 Defensive triangle", value=False,
                                     help="Requires homography calibration")
attacking_team = st.sidebar.selectbox("Attacking team", [0, 1],
                                      format_func=lambda x: f"Team {'A' if x==0 else 'B'}")

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
            from config import (
                DEVICE, OUTPUT_DIR, VIDEO_OUT_DIR, STATS_OUT_DIR, HEAT_OUT_DIR,
                MAX_FRAMES_PANORAMIC, MAX_FRAMES_BROADCAST,
                DET_CONF_PANORAMIC, DET_CONF_BROADCAST,
                BYTETRACK_HIGH_CONF_PANORAMIC, BYTETRACK_HIGH_CONF_BROADCAST,
                REID_ALPHA, MAX_UPLOAD_MB as _MAX_MB, MAX_VIDEO_MINUTES as _MAX_MIN,
            )
            from src.detection import PlayerDetector, detect_video_mode
            from src.tracking import ByteTrackWrapper, run_tracking_loop, run_sort_baseline
            from src.ball import run_ball_detection
            from src.teams import classify_teams
            from src.possession import compute_possession
            from src.heatmap import generate_team_heatmaps, generate_ball_trajectory, generate_possession_chart
            from src.video import render_annotated_video
            from src.evaluation import compare_trackers
            from src.homography import get_homography_hybrid, get_pitch_contour_debug

        fps, img_w, img_h, total_frames, is_panoramic = detect_video_mode(video_path)
        limit_by_minutes = int(MAX_VIDEO_MINUTES * 60 * fps)
        frames_to_run = min(max_frames, total_frames, limit_by_minutes)
        det_conf      = DET_CONF_PANORAMIC if is_panoramic else DET_CONF_BROADCAST
        bt_high_conf  = BYTETRACK_HIGH_CONF_PANORAMIC if is_panoramic else BYTETRACK_HIGH_CONF_BROADCAST
        vname         = Path(video_path).stem

        st.sidebar.info(
            f"**Device:** {DEVICE}\n\n"
            f"**Detection conf:** {det_conf}\n\n"
            f"**ByteTrack high-conf:** {bt_high_conf}"
        )

        # ── Step 1: Detection + Tracking ──────────
        with st.status("🔍 Step 1/5: Detection & Tracking...", expanded=True) as status:
            progress = st.progress(0)
            st.write(f"Processing {frames_to_run} frames with ByteTrack (threshold={bt_high_conf})...")

            def track_progress(idx, total, elapsed):
                pct = idx / total
                eta = (total - idx) / (idx / elapsed) if idx > 0 else 0
                progress.progress(pct, f"Frame {idx}/{total} | ETA: {eta:.0f}s")

            detector  = PlayerDetector(is_panoramic=is_panoramic, device=DEVICE, det_conf=det_conf)
            tracker   = ByteTrackWrapper(fps=fps, activation_threshold=bt_high_conf)
            tracks_df, frames_run, _ = run_tracking_loop(
                video_path, detector, tracker,
                max_frames=frames_to_run, progress_cb=track_progress)

            progress.progress(1.0, "✅ Tracking complete")
            st.write(f"**{tracks_df['track_id'].nunique()} unique player IDs** across {frames_run} frames")
            status.update(label="✅ Step 1: Detection & Tracking complete", state="complete")

        # ── Step 2: Ball Detection ────────────────
        ball_df = None
        if enable_ball:
            with st.status("🏐 Step 2/5: Ball Detection...", expanded=True) as status:
                progress2 = st.progress(0)
                st.write("Using dedicated ball model + Kalman gate filter...")

                def ball_progress(idx, total, elapsed):
                    progress2.progress(idx / total, f"Frame {idx}/{total}")

                ball_df = run_ball_detection(video_path, frames_run,
                                             device=DEVICE, is_panoramic=is_panoramic,
                                             progress_cb=ball_progress)
                detected = ball_df["conf"].notna().sum()
                progress2.progress(1.0, "✅ Ball detection complete")
                st.write(f"**Ball detected in {detected}/{frames_run} frames** ({detected/frames_run*100:.1f}%)")
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
                sort_df, sort_reid_df, reid_available = run_sort_baseline(
                    tracks_df, frames_run, video_path=video_path, device=DEVICE)
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

    if video_path is None:
        st.markdown("---")

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
                    if not br.empty and not np.isnan(br.iloc[0]["cx"]):
                        bx,by = int(br.iloc[0]["cx"]), int(br.iloc[0]["cy"])
                        cv2.circle(ann, (bx,by), 14, (0,255,0), 3)
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

        if st.session_state.get("vid_bytes"):
            st.success(f"Video ready: {st.session_state['vid_filename']}")
            st.download_button(
                label="⬇️ Download rendered video",
                data=st.session_state["vid_bytes"],
                file_name=st.session_state["vid_filename"],
                mime="video/mp4",
                key="dl_btn_cached",
            )
            if st.button("🔄 Re-render video", key="rerender_btn"):
                st.session_state["vid_bytes"]    = None
                st.session_state["vid_filename"] = None
                st.rerun()
        else:
            if _video_path and os.path.exists(str(_video_path)):
                if st.button("🎬 Render full annotated video", key="render_btn"):
                    with st.spinner("Rendering video… this may take a minute"):
                        from src.homography import get_homography_hybrid
                        from src.video import render_annotated_video
                        H, H_inv  = get_homography_hybrid(_video_path)
                        vid_path  = render_annotated_video(
                            _video_path, tracks_df, ball_df or pd.DataFrame(),
                            poss_df, frames_run, _fps, vname,
                            H, H_inv, attacking_team, enable_triangle)
                        with open(vid_path, "rb") as vf:
                            vid_bytes = vf.read()
                        st.session_state["vid_bytes"]    = vid_bytes
                        st.session_state["vid_filename"] = f"{vname}_final.mp4"
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
            comp_df, fig_comp = compare_trackers(tracks_df, sort_df, sort_reid_df, "", frames_run)
            st.dataframe(comp_df, use_container_width=True)
            st.pyplot(fig_comp)
            if not reid_available:
                st.info("Re-ID not available (torchreid not installed). SORT+ReID skipped.")
        else:
            st.info("Enable SORT comparison in sidebar.")
