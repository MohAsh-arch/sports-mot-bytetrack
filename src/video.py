"""
src/video.py — Annotated Video Rendering (Cell 11)
Includes defensive triangle overlay.
"""
import cv2, time
import numpy as np
import pandas as pd
import imageio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEAM_COLORS_BGR, TEAM_NAMES, VIDEO_OUT_DIR
from src.overlays import draw_all_overlays

def render_annotated_video(video_path, tracks_df, ball_df, poss_df,
                           frames_to_run, fps, output_name,
                           H=None, H_inv=None, attacking_team=0,
                           enable_triangle=True,
                           progress_cb=None):
    """
    Render final annotated video with:
      - Team-colored bounding boxes
      - Ball marker
      - Possession bar overlay
    - Defensive triangle (if homography available)
    Returns output path.
    """
    out_path = VIDEO_OUT_DIR / f"{output_name}_final.mp4"

    # Build lookup tables
    frame_tracks = {}
    for _, row in tracks_df.iterrows():
        fid = int(row["frame"])
        frame_tracks.setdefault(fid, []).append(row)

    frame_ball = {}
    for _, row in ball_df.iterrows():
        fid = int(row["frame"])
        if not np.isnan(row["cx"]):
            frame_ball[fid] = (int(row["cx"]), int(row["cy"]))

    frame_poss = {}
    if poss_df is not None:
        for _, row in poss_df.iterrows():
            frame_poss[int(row["frame"])] = row["possession"]

    cap = cv2.VideoCapture(str(video_path))
    frames_buf = []
    idx = 0; poss_accum = []; t0 = time.time()

    while idx < frames_to_run:
        ret, frame = cap.read()
        if not ret: break
        ann = frame.copy()

        # Player boxes
        for row in frame_tracks.get(idx, []):
            tid = int(row["track_id"]); team = int(row["team"])
            is_out = bool(row.get("is_outlier", False))
            color = (128,128,128) if is_out else TEAM_COLORS_BGR.get(team, (128,128,128))
            x1,y1,x2,y2 = int(row["x1"]),int(row["y1"]),int(row["x2"]),int(row["y2"])
            cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
            label = f"#{tid}" if is_out else f"#{tid} {TEAM_NAMES.get(team,'?')}"
            fs = max(0.3, frame.shape[0]/3000)
            cv2.putText(ann, label, (max(0,x1),max(12,y1-4)),
                       cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

        # Ball
        ball = frame_ball.get(idx)
        if ball:
            cv2.circle(ann, ball, 14, (0,255,0), 3)
            cv2.circle(ann, ball, 3, (0,255,0), -1)

        # Geometric overlays
        ann = draw_all_overlays(ann, tracks_df, idx, attacking_team,
                    H, H_inv, enable_triangle)

        # Possession bar
        poss = frame_poss.get(idx)
        if poss in ("Team A","Team B"):
            poss_accum.append(poss)
        ann = _draw_stats_bar(ann, idx, poss_accum, fps)

        frames_buf.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

        if progress_cb and idx % 10 == 0:
            progress_cb(idx, frames_to_run, time.time()-t0)
        idx += 1

    cap.release()

    # Write video
    if frames_buf:
        imageio.mimwrite(str(out_path), frames_buf, fps=int(fps),
                        codec="libx264", quality=7)
    print(f"Video saved: {out_path} ({len(frames_buf)} frames)")
    return out_path

def _draw_stats_bar(frame, frame_idx, poss_accum, fps):
    """Burn possession bar into bottom of frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = max(40, int(h*0.08))
    cv2.rectangle(overlay, (0,h-bar_h), (w,h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if not poss_accum: return frame
    a = sum(1 for p in poss_accum if p=="Team A")
    b = sum(1 for p in poss_accum if p=="Team B")
    total = a + b
    if total == 0: return frame
    a_pct = a/total

    bx1, bx2 = int(w*0.25), int(w*0.75)
    by1, by2 = h-bar_h+8, h-8
    mid = int(bx1 + (bx2-bx1)*a_pct)
    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (50,50,50), -1)
    if mid > bx1: cv2.rectangle(frame, (bx1,by1), (mid,by2), (255,80,20), -1)
    if mid < bx2: cv2.rectangle(frame, (mid,by1), (bx2,by2), (20,20,220), -1)
    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (200,200,200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX; fs = max(0.4, h/2500)
    cv2.putText(frame, f"A {a_pct*100:.0f}%", (bx1-int(w*0.08),by2-2),
               font, fs, (255,180,80), 1, cv2.LINE_AA)
    cv2.putText(frame, f"B {(1-a_pct)*100:.0f}%", (bx2+int(w*0.01),by2-2),
               font, fs, (80,80,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{frame_idx/fps:.1f}s", (12,h-8),
               font, fs*0.9, (180,180,180), 1, cv2.LINE_AA)
    return frame
