"""
src/overlays.py — Defensive Triangle Overlay
Requires homography calibration from src/homography.py.
"""
import cv2
import numpy as np
from src.homography import img_to_pitch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DEFENSIVE_TRI_COLOR, DEFENSIVE_TRI_ALPHA)

def draw_defensive_triangle(frame, tracks_df, frame_idx, attacking_team,
                            H, H_inv):
    """
    Draw triangle between last 3 defenders closest to own goal.
    Semi-transparent orange fill.
    """
    if H is None or H_inv is None:
        return frame

    defending_team = 1 - attacking_team
    defenders = tracks_df[
        (tracks_df["frame"] == frame_idx) &
        (tracks_df["team"] == defending_team) &
        (~tracks_df.get("is_outlier", False))
    ]
    if len(defenders) < 3:
        return frame

    # Sort by pitch-y distance from goal
    pitch_positions = []
    for _, row in defenders.iterrows():
        try:
            pt = img_to_pitch(float(row["cx"]), float(row["y2"]), H)
            pitch_positions.append((float(pt[1]), row))
        except Exception:
            continue

    if len(pitch_positions) < 3:
        return frame

    pitch_positions.sort(key=lambda x: x[0])
    last3 = pitch_positions[-3:]  # 3 closest to own goal

    pts = np.array([
        [int(r["cx"]), int(r["y2"])]
        for _, r in last3
    ], np.int32)

    # Draw triangle outline
    cv2.polylines(frame, [pts.reshape((-1,1,2))], True,
                  DEFENSIVE_TRI_COLOR, 2)

    # Fill with transparent overlay
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts.reshape((-1,1,2))], DEFENSIVE_TRI_COLOR)
    cv2.addWeighted(overlay, DEFENSIVE_TRI_ALPHA, frame,
                    1-DEFENSIVE_TRI_ALPHA, 0, frame)

    return frame

def draw_all_overlays(frame, tracks_df, frame_idx, attacking_team,
                      H, H_inv, enable_triangle=True):
    """Apply geometric overlays to a frame."""
    if enable_triangle:
        frame = draw_defensive_triangle(frame, tracks_df, frame_idx,
                                        attacking_team, H, H_inv)
    return frame
