"""
src/tracking.py — ByteTrack + SORT Trackers
FIX #1: ByteTrack activation threshold = 0.35
"""

import cv2, time, warnings
import numpy as np
import pandas as pd
import supervision as sv
from pathlib import Path
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE,
    BYTETRACK_ACTIVATION_THRESHOLD, BYTETRACK_LOST_BUFFER,
    BYTETRACK_MATCH_THRESHOLD, SORT_MAX_AGE, SORT_MIN_HITS,
    SORT_IOU_THRESH, REID_ALPHA,
)

def iou(bb_a, bb_b):
    xx1, yy1 = max(bb_a[0], bb_b[0]), max(bb_a[1], bb_b[1])
    xx2, yy2 = min(bb_a[2], bb_b[2]), min(bb_a[3], bb_b[3])
    w, h = max(0., xx2-xx1), max(0., yy2-yy1)
    inter = w * h
    area_a = (bb_a[2]-bb_a[0]) * (bb_a[3]-bb_a[1])
    area_b = (bb_b[2]-bb_b[0]) * (bb_b[3]-bb_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.

class ByteTrackWrapper:
    """sv.ByteTrack with corrected activation threshold split."""
    def __init__(self, fps=25, activation_threshold=None):
        activation_threshold = (
            BYTETRACK_ACTIVATION_THRESHOLD
            if activation_threshold is None
            else activation_threshold
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = sv.ByteTrack(
                track_activation_threshold=activation_threshold,
                lost_track_buffer=BYTETRACK_LOST_BUFFER,
                minimum_matching_threshold=BYTETRACK_MATCH_THRESHOLD,
                frame_rate=fps,
            )
        print(f"ByteTrack: activation={activation_threshold}, "
              f"buffer={BYTETRACK_LOST_BUFFER}")

    def reset(self):
        self.tracker.reset()

    def update(self, detections):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return self.tracker.update_with_detections(detections)

class KalmanBoxTracker:
    _count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7, dtype=np.float32)
        for i in range(4): self.kf.F[i, i+3] = 1.
        self.kf.H = np.eye(4, 7, dtype=np.float32)
        self.kf.R = np.eye(4, dtype=np.float32)
        self.kf.R[2:,2:] *= 10.
        self.kf.P = np.eye(7, dtype=np.float32)
        self.kf.P[4:,4:] *= 1000.; self.kf.P *= 10.
        self.kf.Q = np.eye(7, dtype=np.float32)
        self.kf.Q[-1,-1] *= 0.01; self.kf.Q[4:,4:] *= 0.01
        x1,y1,x2,y2 = [float(v) for v in bbox[:4]]
        cx,cy = (x1+x2)/2, (y1+y2)/2
        w,h = x2-x1, y2-y1
        self.kf.x = np.array([[cx],[cy],[w*h],[w/(h+1e-6)],[0.],[0.],[0.]], dtype=np.float32)
        self.time_since_update = 0
        self.id = KalmanBoxTracker._count; KalmanBoxTracker._count += 1
        self.hit_streak = 0

    def update(self, bbox):
        x1,y1,x2,y2 = [float(v) for v in bbox[:4]]
        cx,cy = (x1+x2)/2, (y1+y2)/2
        w,h = x2-x1, y2-y1
        self.kf.update(np.array([[cx],[cy],[w*h],[w/(h+1e-6)]], dtype=np.float32))
        self.time_since_update = 0; self.hit_streak += 1

    def predict(self):
        self.kf.predict(); self.time_since_update += 1
        return self._state()

    def _state(self):
        x = self.kf.x.flatten()
        cx,cy,s,r = float(x[0]),float(x[1]),float(x[2]),float(x[3])
        w = float(np.sqrt(max(s*r,1e-6)))
        h = float(np.sqrt(max(s/max(r,1e-6),1e-6)))
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

class SORT:
    """IoU-only tracker. No low-conf rescue → more ID switches."""
    def __init__(self, max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                 iou_thresh=SORT_IOU_THRESH):
        self.max_age, self.min_hits, self.iou_thresh = max_age, min_hits, iou_thresh
        self.trackers = []; self.frame_count = 0
        KalmanBoxTracker._count = 0

    def update(self, dets):
        self.frame_count += 1; results = []
        predicted = [t.predict() for t in self.trackers]
        if self.trackers and dets:
            iou_mat = np.zeros((len(dets), len(predicted)), dtype=np.float32)
            for d,det in enumerate(dets):
                for t,pred in enumerate(predicted):
                    iou_mat[d,t] = iou(det, pred)
            ri, ci = linear_sum_assignment(-iou_mat)
            matched_d, matched_t = set(), set()
            for r,c in zip(ri, ci):
                if iou_mat[r,c] >= self.iou_thresh:
                    self.trackers[c].update(dets[r]); matched_d.add(r); matched_t.add(c)
            for d,det in enumerate(dets):
                if d not in matched_d: self.trackers.append(KalmanBoxTracker(det))
        else:
            for det in dets: self.trackers.append(KalmanBoxTracker(det))
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        for t in self.trackers:
            if t.time_since_update == 0 and t.hit_streak >= self.min_hits:
                results.append(t._state() + [t.id + 1])
        return results

def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b))

class SORTWithReID:
    """
    SORT tracker augmented with optional Re-ID embedding.
    cost = alpha * cosine_dist + (1 - alpha) * (1 - IoU)
    """
    def __init__(self, max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                 iou_thresh=SORT_IOU_THRESH, reid_map=None, alpha=0.0):
        self.max_age, self.min_hits, self.iou_thresh = max_age, min_hits, iou_thresh
        self.trackers = []; self.frame_count = 0
        self.reid_map = reid_map or {}
        self.alpha = alpha
        self.id_to_reid = {}
        KalmanBoxTracker._count = 0

    def update(self, dets, det_ids=None):
        self.frame_count += 1; results = []
        predicted = [t.predict() for t in self.trackers]

        if self.trackers and dets:
            cost_mat = np.zeros((len(dets), len(predicted)), dtype=np.float32)
            for d, det in enumerate(dets):
                for t, pred in enumerate(predicted):
                    iou_cost = 1.0 - iou(det, pred)
                    reid_cost = 0.5
                    if self.alpha > 0 and det_ids is not None:
                        did = det_ids[d]
                        tid_internal = self.trackers[t].id + 1
                        d_feat = self.reid_map.get(did)
                        t_feat = self.id_to_reid.get(tid_internal)
                        if d_feat is not None and t_feat is not None:
                            reid_cost = cosine_dist(d_feat, t_feat)
                    cost_mat[d, t] = self.alpha * reid_cost + (1 - self.alpha) * iou_cost

            ri, ci = linear_sum_assignment(cost_mat)
            matched_d, matched_t = set(), set()
            for r, c in zip(ri, ci):
                if cost_mat[r, c] < (1 - self.iou_thresh):
                    self.trackers[c].update(dets[r]); matched_d.add(r); matched_t.add(c)
                    if det_ids is not None and det_ids[r] in self.reid_map:
                        self.id_to_reid[self.trackers[c].id + 1] = self.reid_map[det_ids[r]]

            for d, det in enumerate(dets):
                if d not in matched_d:
                    self.trackers.append(KalmanBoxTracker(det))
        else:
            for det in dets:
                self.trackers.append(KalmanBoxTracker(det))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        for t in self.trackers:
            if t.time_since_update == 0 and t.hit_streak >= self.min_hits:
                results.append(t._state() + [t.id + 1])
        return results

def build_reid_map(video_path, tracks_df, device=DEVICE):
    """Build per-track Re-ID embeddings using OSNet-AIN. Returns (map, available)."""
    try:
        import torch
        import torchreid
        import torchvision.transforms as T
    except Exception:
        return {}, False

    model = torchreid.models.build_model(
        name="osnet_ain_x1_0", num_classes=1000, pretrained=True
    )
    model.eval()
    if device == "cuda":
        model = model.cuda()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = tracks_df
    if "is_outlier" in df.columns:
        df = df[~df["is_outlier"]]
    best_frames = (
        df.sort_values("conf", ascending=False)
          .groupby("track_id").first().reset_index()
    )

    reid_map = {}
    cap = cv2.VideoCapture(str(video_path))
    prev_fidx = -1; cur_frame = None
    for _, row in best_frames.iterrows():
        fidx = int(row["frame"])
        if fidx != prev_fidx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, cur_frame = cap.read()
            if not ret:
                continue
            prev_fidx = fidx

        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        h_f, w_f = cur_frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w_f, x2); y2 = min(h_f, y2)
        if (y2 - y1) < 20 or (x2 - x1) < 10:
            continue
        crop = cur_frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0)
        if device == "cuda":
            tensor = tensor.cuda()
        with torch.no_grad():
            feat = model(tensor)
        reid_map[int(row["track_id"])] = feat.cpu().numpy().flatten()

    cap.release()
    return reid_map, True

def run_tracking_loop(video_path, detector, tracker, max_frames=None, progress_cb=None):
    """Run detection + ByteTrack. Returns (tracks_df, frames_run, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frames_to_run = min(max_frames, total) if max_frames else total
    tracker.reset(); rows = []; idx = 0; t0 = time.time()
    try:
        while idx < frames_to_run:
            ret, frame = cap.read()
            if not ret: break
            dets = detector.detect(frame)
            tracked = tracker.update(dets)
            for i in range(len(tracked)):
                x1,y1,x2,y2 = tracked.xyxy[i]
                rows.append({"frame":idx,"track_id":int(tracked.tracker_id[i]),
                    "x1":round(float(x1),1),"y1":round(float(y1),1),
                    "x2":round(float(x2),1),"y2":round(float(y2),1),
                    "cx":round(float((x1+x2)/2),1),"cy":round(float((y1+y2)/2),1),
                    "conf":round(float(tracked.confidence[i]),3),
                    "width":round(float(x2-x1),1),"height":round(float(y2-y1),1)})
            if progress_cb and idx % 5 == 0:
                progress_cb(idx, frames_to_run, time.time()-t0)
            idx += 1
    finally:
        cap.release()
    df = pd.DataFrame(rows)
    print(f"ByteTrack: {idx} frames, {df['track_id'].nunique()} IDs, "
          f"{time.time()-t0:.1f}s")
    return df, idx, fps

def run_sort_baseline(tracks_df, frames_to_run, video_path=None,
                      device=DEVICE, reid_alpha=REID_ALPHA):
    """
    Run SORT on same detections for comparison.
    Returns (sort_df, sort_reid_df, reid_available).
    """
    KalmanBoxTracker._count = 0
    s = SORT(); rows = []
    for fid in range(frames_to_run):
        fd = tracks_df[tracks_df["frame"]==fid]
        dets = fd[["x1","y1","x2","y2"]].values.tolist()
        out = s.update(dets)
        for t in out:
            rows.append({"frame":fid,"track_id":int(t[4]),
                "x1":round(t[0],1),"y1":round(t[1],1),
                "x2":round(t[2],1),"y2":round(t[3],1),
                "cx":round((t[0]+t[2])/2,1),"cy":round((t[1]+t[3])/2,1),
                "width":round(t[2]-t[0],1)})
    sort_df = pd.DataFrame(rows)

    sort_reid_df = None
    reid_map = {}
    reid_available = False
    if video_path:
        reid_map, reid_available = build_reid_map(video_path, tracks_df, device=device)

    if reid_available and reid_alpha > 0:
        KalmanBoxTracker._count = 0
        s_reid = SORTWithReID(reid_map=reid_map, alpha=reid_alpha)
        rows = []
        for fid in range(frames_to_run):
            fd = tracks_df[tracks_df["frame"]==fid]
            dets = fd[["x1","y1","x2","y2"]].values.tolist()
            dids = fd["track_id"].tolist()
            out = s_reid.update(dets, dids)
            for t in out:
                rows.append({"frame":fid,"track_id":int(t[4]),
                    "x1":round(t[0],1),"y1":round(t[1],1),
                    "x2":round(t[2],1),"y2":round(t[3],1),
                    "cx":round((t[0]+t[2])/2,1),"cy":round((t[1]+t[3])/2,1),
                    "width":round(t[2]-t[0],1)})
        sort_reid_df = pd.DataFrame(rows)

    print(f"SORT: {sort_df['track_id'].nunique()} IDs vs ByteTrack: "
          f"{tracks_df['track_id'].nunique()} IDs")
    return sort_df, sort_reid_df, reid_available
