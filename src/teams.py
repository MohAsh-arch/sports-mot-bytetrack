"""
src/teams.py — Overlap-Aware Team Classification (Fix #4)
Builds team centroids from NON-overlapping frames only,
then classifies all frames via nearest-centroid.
"""
import cv2, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (N_TEAMS, MIN_CROP_PX, KMEANS_SEED, OVERLAP_IOU_THRESH)
from src.tracking import iou

def extract_jersey_feature(frame, x1, y1, x2, y2):
    """Extract RGB+HSV combined feature from torso crop."""
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    h_f, w_f = frame.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w_f,x2), min(h_f,y2)
    bh, bw = y2-y1, x2-x1
    if bh < MIN_CROP_PX or bw < MIN_CROP_PX:
        return None
    ty1 = y1 + int(bh*0.25)
    ty2 = y1 + int(bh*0.65)
    crop = frame[ty1:ty2, x1:x2]
    if crop.size == 0:
        return None
    rgb_mean = crop.mean(axis=(0,1))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.array([rgb_mean[2], rgb_mean[1], rgb_mean[0],
                     hsv[:,:,0].mean(), hsv[:,:,1].mean(),
                     hsv[:,:,2].mean()], dtype=np.float32)

def find_overlapping_ids(tracks_df, frame_idx, thresh=OVERLAP_IOU_THRESH):
    """Find track IDs that overlap in a given frame."""
    group = tracks_df[tracks_df["frame"]==frame_idx]
    if len(group) < 2:
        return set()
    boxes = group[["x1","y1","x2","y2"]].values
    ids = group["track_id"].values
    overlapping = set()
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if iou(boxes[i], boxes[j]) > thresh:
                overlapping.add(ids[i]); overlapping.add(ids[j])
    return overlapping

def classify_teams(tracks_df, video_path, progress_cb=None):
    """
    Overlap-aware team classification.
    1. Build per-ID features from non-overlapping frames
    2. Average features per ID
    3. KMeans on clean features
    4. Assign all tracks via nearest centroid
    Returns (tracks_df with 'team' column, team_map dict).
    """
    cap = cv2.VideoCapture(str(video_path))
    per_id_feats = defaultdict(list)

    prev_fidx = -1; cur_frame = None
    frame_groups = sorted(tracks_df.groupby("frame"), key=lambda x: x[0])
    total_frames = len(frame_groups)

    for idx, (fidx, group) in enumerate(frame_groups):
        if fidx != prev_fidx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, cur_frame = cap.read()
            if not ret:
                continue
            prev_fidx = fidx

        boxes = group[["x1","y1","x2","y2"]].values
        ids = group["track_id"].values
        overlapping = set()
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if iou(boxes[i], boxes[j]) > OVERLAP_IOU_THRESH:
                    overlapping.add(ids[i]); overlapping.add(ids[j])

        for _, row in group.iterrows():
            tid = int(row["track_id"])
            if tid in overlapping:
                continue
            feat = extract_jersey_feature(cur_frame, row["x1"], row["y1"],
                                          row["x2"], row["y2"])
            if feat is not None:
                per_id_feats[tid].append(feat)

        if progress_cb and idx % 20 == 0:
            progress_cb(idx, total_frames, 0)

    cap.release()

    valid_ids = []
    features = []
    for tid, feats in per_id_feats.items():
        if len(feats) > 0:
            valid_ids.append(tid)
            features.append(np.mean(feats, axis=0))

    if len(features) < N_TEAMS:
        print(f"Warning: only {len(features)} IDs with clean crops, need {N_TEAMS}")
        tracks_df["team"] = -1
        return tracks_df, {}

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_inertia = float("inf"); best_labels = None
    for seed in range(10):
        km = KMeans(n_clusters=N_TEAMS, random_state=seed, n_init=20)
        labels = km.fit_predict(X_scaled)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_; best_labels = labels

    counts = Counter(best_labels)
    if counts.get(0,0) < counts.get(1,0):
        best_labels = 1 - best_labels

    team_map = {valid_ids[i]: int(best_labels[i]) for i in range(len(valid_ids))}
    for tid in tracks_df["track_id"].unique():
        if tid not in team_map:
            team_map[tid] = -1

    tracks_df["team"] = tracks_df["track_id"].map(team_map).fillna(-1).astype(int)

    # Outlier filtering (referee/GK detection)
    outlier_ids = set()
    for team_id in [0, 1]:
        idxs = [i for i,tid in enumerate(valid_ids) if team_map.get(tid,-1)==team_id]
        if not idxs: continue
        tf = X_scaled[idxs]
        centroid = tf.mean(axis=0, keepdims=True)
        dists = cdist(tf, centroid, metric="euclidean").flatten()
        thresh = dists.mean() + 1.5 * dists.std()
        for idx, dist in zip(idxs, dists):
            if dist > thresh:
                outlier_ids.add(valid_ids[idx])

    tracks_df["is_outlier"] = tracks_df["track_id"].isin(outlier_ids)

    print(f"Teams: A={counts.get(0,0)}, B={counts.get(1,0)}, "
          f"outliers={len(outlier_ids)}")
    return tracks_df, team_map
