"""
src/heatmap.py — KDE Heatmaps + Ball Trajectory (Cell 10)
Ball trajectory now uses Kalman-filtered positions (no ghost dots).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HEATMAP_RESOLUTION, KDE_BANDWIDTH, ALPHA_HEATMAP

def draw_pitch(ax, pitch_w=6500, pitch_h=1000, color="white"):
    """Draw simplified top-down panoramic pitch."""
    ax.set_facecolor("#2d8a4e")
    lw, pw, ph = 2.5, pitch_w, pitch_h
    ax.plot([0,pw,pw,0,0], [0,0,ph,ph,0], color=color, lw=lw)
    ax.plot([pw/2,pw/2], [0,ph], color=color, lw=lw)
    cc = Circle((pw/2,ph/2), radius=ph*0.18, fill=False, color=color, lw=lw)
    ax.add_patch(cc)
    ax.plot(pw/2, ph/2, "o", color=color, markersize=4)
    # Penalty boxes
    pb_w, pb_h = pw*0.12, ph*0.6
    pb_y = (ph-pb_h)/2
    ax.plot([0,pb_w,pb_w,0], [pb_y,pb_y,pb_y+pb_h,pb_y+pb_h], color=color, lw=lw)
    ax.plot([pw,pw-pb_w,pw-pb_w,pw], [pb_y,pb_y,pb_y+pb_h,pb_y+pb_h], color=color, lw=lw)
    # Goal boxes
    gb_w, gb_h = pw*0.04, ph*0.28
    gb_y = (ph-gb_h)/2
    ax.plot([0,gb_w,gb_w,0], [gb_y,gb_y,gb_y+gb_h,gb_y+gb_h], color=color, lw=lw)
    ax.plot([pw,pw-gb_w,pw-gb_w,pw], [gb_y,gb_y,gb_y+gb_h,gb_y+gb_h], color=color, lw=lw)
    ax.set_xlim(0,pw); ax.set_ylim(0,ph); ax.set_aspect("equal"); ax.axis("off")

def compute_kde(x_vals, y_vals, img_w, img_h, grid_w, grid_h, bw=KDE_BANDWIDTH):
    if img_w <= 0 or img_h <= 0:
        return np.zeros((grid_h, grid_w))
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid]; y_vals = y_vals[valid]
    if len(x_vals) < 5:
        return np.zeros((grid_h, grid_w))
    x_n = np.array(x_vals)/img_w; y_n = np.array(y_vals)/img_h
    xy = np.vstack([x_n, y_n])
    kde = gaussian_kde(xy, bw_method=bw)
    xi = np.linspace(0,1,grid_w); yi = np.linspace(0,1,grid_h)
    Xi, Yi = np.meshgrid(xi, yi)
    return kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(grid_h, grid_w)

def generate_team_heatmaps(tracks_df, img_w, img_h):
    """Generate team heatmap figure. Returns matplotlib figure."""
    if tracks_df is None or len(tracks_df) == 0:
        fig, ax = plt.subplots(figsize=(18, 4))
        draw_pitch(ax, img_w, img_h)
        ax.set_title("Player Position Heatmaps — no tracking data", fontsize=13)
        return fig

    clean = tracks_df[~tracks_df.get("is_outlier", False)] if "is_outlier" in tracks_df.columns else tracks_df
    ta = clean[clean["team"]==0]; tb = clean[clean["team"]==1]
    gw, gh = HEATMAP_RESOLUTION

    cmap_a = LinearSegmentedColormap.from_list("ta", ["#000080","#0000FF","#00FFFF","#FFFF00"], N=256)
    cmap_b = LinearSegmentedColormap.from_list("tb", ["#800000","#FF0000","#FF8000","#FFFF00"], N=256)

    Za = compute_kde(ta["cx"], ta["cy"], img_w, img_h, gw, gh)
    Zb = compute_kde(tb["cx"], tb["cy"], img_w, img_h, gw, gh)

    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    fig.suptitle("Player Position Heatmaps", fontsize=14, fontweight="bold")

    for ax, Z, cmap, title in [
        (axes[0], Za, cmap_a, "Team A"),
        (axes[1], Zb, cmap_b, "Team B"),
    ]:
        draw_pitch(ax, img_w, img_h)
        Zp = Z/Z.max() if Z.max()>0 else Z
        ax.imshow(Zp, extent=[0,img_w,img_h,0], origin="upper",
                  cmap=cmap, alpha=ALPHA_HEATMAP, aspect="auto", interpolation="bilinear")
        ax.set_title(title, fontsize=12, pad=8)

    plt.tight_layout()
    return fig

def generate_ball_trajectory(ball_df, frames_to_run, fps, img_w, img_h):
    """Generate ball trajectory figure. Returns matplotlib figure."""
    if ball_df is None or len(ball_df) == 0:
        fig, ax = plt.subplots(figsize=(18, 4))
        draw_pitch(ax, img_w, img_h)
        ax.set_title("Ball Trajectory — no detections", fontsize=13)
        return fig

    ball_valid = ball_df[ball_df["conf"].notna()].sort_values("frame")
    all_frames = pd.DataFrame({"frame": range(frames_to_run)})
    bi = all_frames.merge(ball_valid, on="frame", how="left")
    bi["cx"] = bi["cx"].interpolate(method="linear")
    bi["cy"] = bi["cy"].interpolate(method="linear")
    bi = bi.dropna(subset=["cx","cy"])

    if len(bi) < 2:
        fig, ax = plt.subplots(figsize=(18, 4))
        draw_pitch(ax, img_w, img_h)
        ax.set_title("Ball Trajectory — insufficient data", fontsize=13)
        return fig

    fig, ax = plt.subplots(figsize=(18, 4))
    draw_pitch(ax, img_w, img_h)

    points = np.array([bi["cx"], bi["cy"]]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(bi))
    lc = LineCollection(segments, cmap=plt.cm.plasma, norm=norm, linewidth=2, alpha=0.85)
    lc.set_array(np.arange(len(bi)))
    ax.add_collection(lc)

    ax.plot(float(bi.iloc[0]["cx"]), float(bi.iloc[0]["cy"]), "o",
            color="white", markersize=8, zorder=5, label="Start")
    ax.plot(float(bi.iloc[-1]["cx"]), float(bi.iloc[-1]["cy"]), "s",
            color="yellow", markersize=8, zorder=5, label="End")

    cbar = plt.colorbar(lc, ax=ax, orientation="horizontal", pad=0.02, fraction=0.03)
    cbar.set_label("Frame (time →)", fontsize=9)
    ax.set_title("Ball Trajectory (Kalman-filtered)", fontsize=13, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig

def generate_possession_chart(poss_df, a_pct, b_pct, fps):
    """Generate possession bar + timeline chart. Returns matplotlib figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ball Possession Analysis", fontsize=14)

    if poss_df is None or poss_df.empty or fps <= 0:
        for ax in axes: ax.axis("off")
        axes[0].set_title("Insufficient Data", fontsize=12)
        return fig

    ax1 = axes[0]
    bars = ax1.bar(["Team A","Team B"], [a_pct, b_pct],
                   color=["#2563EB","#DC2626"], width=0.5, edgecolor="white", linewidth=1.5)
    ax1.set_ylim(0,100); ax1.set_ylabel("Possession (%)", fontsize=11)
    ax1.set_title("Ball Possession %", fontsize=12)
    ax1.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar,pct in zip(bars, [a_pct,b_pct]):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax1.spines[["top","right"]].set_visible(False)

    ax2 = axes[1]
    ps = poss_df["possession"].map({"Team A":1,"Team B":0,"contested":np.nan,"no_data":np.nan})
    rp = ps.rolling(window=25, min_periods=5).mean() * 100
    fs = poss_df["frame"] / fps
    if not np.isfinite(rp).any():
        for ax in axes: ax.axis("off")
        axes[0].set_title("Insufficient Data", fontsize=12)
        return fig
    ax2.fill_between(fs, rp, 50, where=(rp>=50), color="#2563EB", alpha=0.4, label="Team A")
    ax2.fill_between(fs, rp, 50, where=(rp<50), color="#DC2626", alpha=0.4, label="Team B")
    ax2.plot(fs, rp, color="black", linewidth=1.2, alpha=0.7)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Team A %"); ax2.set_ylim(0,100)
    ax2.set_title("Possession Over Time"); ax2.legend(fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    return fig
