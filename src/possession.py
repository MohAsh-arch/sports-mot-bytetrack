"""
src/possession.py — Ball Possession Analysis (Cell 9)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_POSSESSION_DIST

def compute_possession(tracks_df, ball_df):
    """
    Per frame: find closest player to ball, assign possession to their team.
    Returns possession_df with columns:
      frame, ball_cx, ball_cy, closest_id, closest_dist, team, possession
    """
    ball_detected = ball_df[ball_df["conf"].notna()].copy()
    rows = []

    for _, ball_row in ball_detected.iterrows():
        fid = int(ball_row["frame"])
        bcx, bcy = ball_row["cx"], ball_row["cy"]

        players = tracks_df[tracks_df["frame"] == fid]
        if "is_outlier" in players.columns:
            players = players[~players["is_outlier"]]

        if players.empty:
            rows.append({"frame":fid, "ball_cx":bcx, "ball_cy":bcy,
                        "closest_id":-1, "closest_dist":float("nan"),
                        "team":-1, "possession":"no_data"})
            continue

        dx = players["cx"] - bcx
        dy = players["cy"] - bcy
        dist = np.sqrt(dx**2 + dy**2)
        min_idx = dist.idxmin()
        min_dist = dist[min_idx]
        closest = players.loc[min_idx]

        if min_dist <= MAX_POSSESSION_DIST:
            team = int(closest["team"])
            poss = {0:"Team A", 1:"Team B"}.get(team, "unknown")
        else:
            team = -1; poss = "contested"

        rows.append({"frame":fid, "ball_cx":bcx, "ball_cy":bcy,
                    "closest_id":int(closest["track_id"]),
                    "closest_dist":round(float(min_dist),1),
                    "team":team, "possession":poss})

    poss_df = pd.DataFrame(rows)
    if poss_df.empty:
        poss_df = pd.DataFrame(columns=["frame", "ball_cx", "ball_cy", "closest_id", "closest_dist", "team", "possession"])
        return poss_df, 0.0, 0.0

    # Compute percentages
    counts = poss_df["possession"].value_counts()
    a = counts.get("Team A", 0)
    b = counts.get("Team B", 0)
    assigned = a + b
    if assigned > 0:
        a_pct = a / assigned * 100
        b_pct = b / assigned * 100
    else:
        a_pct = b_pct = 0.0

    print(f"Possession: Team A {a_pct:.1f}%, Team B {b_pct:.1f}%")
    return poss_df, a_pct, b_pct
