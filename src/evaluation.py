"""
src/evaluation.py — MOT Metrics + ByteTrack vs SORT Comparison (Cell 13)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def compute_mot_metrics(tracks_df, gt_file, frames_to_run, label="Tracker"):
    """
    Compute proxy MOT metrics by comparing tracker output to GT.

    Supports two GT formats (auto-detected by column count):
      • Standard MOT (9 cols):  frame,id,x,y,w,h,conf,cls,vis
      • TeamTrack   (10 cols):  frame,id,x,y,w,h,conf,x_world,y_world,z_world
    """
    gt_path = Path(gt_file) if gt_file else None
    if not gt_path or not gt_path.is_file():
        return None

    # Peek at the first data row to count columns
    with open(gt_path) as _f:
        _first = _f.readline().strip()
    _ncols = len(_first.split(","))

    if _ncols >= 10:
        # TeamTrack format (10+ columns)
        _names = ["frame","id","x","y","w","h","conf","x_world","y_world","z_world"]
    else:
        # Standard MOT format (9 columns)
        _names = ["frame","id","x","y","w","h","conf","cls","vis"]

    gt = pd.read_csv(gt_path, header=None, names=_names[:_ncols])
    gt = gt[gt["conf"]==1].copy()
    gt["x2"] = gt["x"]+gt["w"]; gt["y2"] = gt["y"]+gt["h"]
    gt["cx"] = gt["x"]+gt["w"]/2; gt["cy"] = gt["y"]+gt["h"]/2
    gt["frame"] = gt["frame"] - 1  # GT is 1-indexed
    gt = gt[gt["frame"] < frames_to_run]

    total_gt = total_det = total_matches = id_switches = 0
    prev_matches = {}

    for fid in range(frames_to_run):
        gf = gt[gt["frame"]==fid]; df = tracks_df[tracks_df["frame"]==fid]
        total_gt += len(gf); total_det += len(df)
        if gf.empty or df.empty:
            prev_matches = {}; continue

        curr_matches = {}; used = set()
        for _, g in gf.iterrows():
            dx = df["cx"]-g["cx"]; dy = df["cy"]-g["cy"]
            dist = np.sqrt(dx**2+dy**2)
            bi = dist.idxmin(); bd = dist[bi]
            thresh = df.loc[bi,"width"]*1.5 if "width" in df.columns else 80
            if bd < thresh and bi not in used:
                det_id = int(df.loc[bi,"track_id"])
                curr_matches[int(g["id"])] = det_id; used.add(bi)
                total_matches += 1
                if int(g["id"]) in prev_matches and prev_matches[int(g["id"])] != det_id:
                    id_switches += 1
        prev_matches = curr_matches

    prec = total_matches/total_det if total_det > 0 else 0
    rec = total_matches/total_gt if total_gt > 0 else 0
    fp = total_det - total_matches; fn = total_gt - total_matches
    mota = 1 - (fn+fp+id_switches)/total_gt if total_gt > 0 else 0
    idf1 = 2*total_matches/(total_gt+total_det) if (total_gt+total_det)>0 else 0

    return {
        "Tracker": label,
        "MOTA": round(mota*100, 2), "IDF1": round(idf1*100, 2),
        "Precision": round(prec*100, 2), "Recall": round(rec*100, 2),
        "ID Switches": id_switches, "GT Objects": total_gt,
        "Detections": total_det, "Matches": total_matches,
    }

def compare_trackers(byte_df, sort_df, sort_reid_df, gt_file, frames_to_run):
    """
    Compare ByteTrack vs SORT (+ optional SORT+ReID).
    Returns (results_df, figure) or (None, None).
    """
    if frames_to_run <= 0:
        stats = {
            "Metric": ["Unique IDs", "Avg tracks/frame", "Total detections"],
            "ByteTrack": [0, 0, 0],
            "SORT": [0, 0, 0],
        }
        if sort_reid_df is not None:
            stats["SORT+ReID"] = [0, 0, 0]
        return pd.DataFrame(stats), _make_stats_chart(stats)

    mb = compute_mot_metrics(byte_df, gt_file, frames_to_run, "ByteTrack")
    ms = compute_mot_metrics(sort_df, gt_file, frames_to_run, "SORT")
    mr = None
    if sort_reid_df is not None:
        mr = compute_mot_metrics(sort_reid_df, gt_file, frames_to_run, "SORT+ReID")

    if mb is None or ms is None:
        stats = {
            "Metric": ["Unique IDs", "Avg tracks/frame", "Total detections"],
            "ByteTrack": [byte_df["track_id"].nunique(),
                         round(len(byte_df)/frames_to_run, 1), len(byte_df)],
            "SORT": [sort_df["track_id"].nunique(),
                    round(len(sort_df)/frames_to_run, 1), len(sort_df)],
        }
        if sort_reid_df is not None:
            stats["SORT+ReID"] = [sort_reid_df["track_id"].nunique(),
                                  round(len(sort_reid_df)/frames_to_run, 1),
                                  len(sort_reid_df)]
        return pd.DataFrame(stats), _make_stats_chart(stats)

    results = [mb, ms] + ([mr] if mr is not None else [])
    results_df = pd.DataFrame(results)

    metrics = ["MOTA","IDF1","Precision","Recall"]
    x = np.arange(len(metrics))
    width = 0.25 if mr is not None else 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    bars = []
    bars.append(ax.bar(x - width, [mb[m] for m in metrics], width,
                       label="ByteTrack", color="#2563EB", alpha=0.85))
    bars.append(ax.bar(x, [ms[m] for m in metrics], width,
                       label="SORT", color="#DC2626", alpha=0.85))
    if mr is not None:
        bars.append(ax.bar(x + width, [mr[m] for m in metrics], width,
                           label="SORT+ReID", color="#16A34A", alpha=0.85))

    ax.set_ylabel("Score (%)")
    ax.set_title("Tracker Comparison — ByteTrack vs SORT", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0,110)
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    for bar_group in bars:
        for bar in bar_group:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return results_df, fig

def _make_stats_chart(stats):
    """Simple comparison chart when no GT available."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stats["Metric"]))
    w = 0.25 if "SORT+ReID" in stats else 0.35
    ax.bar(x-w, stats["ByteTrack"], w, label="ByteTrack", color="#2563EB")
    ax.bar(x, stats["SORT"], w, label="SORT", color="#DC2626")
    if "SORT+ReID" in stats:
        ax.bar(x+w, stats["SORT+ReID"], w, label="SORT+ReID", color="#16A34A")
    ax.set_xticks(x); ax.set_xticklabels(stats["Metric"])
    ax.set_title("ByteTrack vs SORT — Statistics"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig
