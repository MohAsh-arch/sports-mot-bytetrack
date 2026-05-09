"""
src/homography.py — Pitch Keypoint Calibration (Fix #3 support)
Maps image pixels ↔ real-world pitch coordinates (metres).
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PITCH_LENGTH, PITCH_WIDTH, PROJECT_ROOT

KEYPOINTS_FILE = PROJECT_ROOT / "pitch_keypoints.json"

def img_to_pitch(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

def pitch_to_img(rx, ry, H_inv):
    pt = np.array([[[rx, ry]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H_inv)[0][0].astype(int)

def calibrate_homography(src_pts, dst_pts):
    """
    Compute homography from image points to pitch coordinates.
    src_pts: Nx2 array of pixel coords  (at least 4 points)
    dst_pts: Nx2 array of pitch coords in metres
    Returns (H, H_inv) or (None, None) on failure.
    """
    src = np.float32(src_pts)
    dst = np.float32(dst_pts)
    if len(src) < 4:
        print("Need at least 4 keypoints for homography")
        return None, None
    H, status = cv2.findHomography(src, dst)
    if H is None:
        return None, None
    H_inv = np.linalg.inv(H)
    return H, H_inv

def save_keypoints(src_pts, dst_pts, path=None):
    """Save keypoints to JSON for reuse."""
    path = path or KEYPOINTS_FILE
    data = {
        "src_pts": np.array(src_pts).tolist(),
        "dst_pts": np.array(dst_pts).tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Keypoints saved to {path}")

def load_keypoints(path=None):
    """Load keypoints from JSON. Returns (src_pts, dst_pts) or (None, None)."""
    path = path or KEYPOINTS_FILE
    if not Path(path).exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    return np.float32(data["src_pts"]), np.float32(data["dst_pts"])

def get_homography():
    """
    Try to load saved keypoints and compute homography.
    Returns (H, H_inv) or (None, None) if not calibrated.
    """
    src, dst = load_keypoints()
    if src is None:
        return None, None
    return calibrate_homography(src, dst)

def _order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected 4x2 points")
    # sort by y, then x to split top/bottom
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bottom = pts[2:][np.argsort(pts[2:, 0])[::-1]]
    return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

def _detect_pitch_corners(frame):
    """Detect 4 pitch corners from a frame. Returns (src_pts, contour) or (None, None)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_lower = np.array([25, 30, 40])
    green_upper = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = None
    for eps_factor in [0.02, 0.05, 0.1]:
        cand = cv2.approxPolyDP(largest, eps_factor * peri, True)
        if len(cand) == 4:
            approx = cand
            break

    if approx is None:
        hull = cv2.convexHull(largest)
        cand = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(cand) == 4:
            approx = cand
        else:
            rect = cv2.minAreaRect(hull)
            approx = cv2.boxPoints(rect).astype(np.int32).reshape(4, 1, 2)

    try:
        src_pts = _order_corners(approx[:, 0, :])
    except Exception:
        return None, None

    return src_pts, largest

def auto_homography_from_frame(frame):
    """
    Auto-calibrate homography from pitch contour in a frame.
    Returns (H, H_inv, ok).
    """
    if frame is None or frame.size == 0:
        return None, None, False

    src_pts, _ = _detect_pitch_corners(frame)
    if src_pts is None:
        return None, None, False

    dst_pts = np.float32([
        [0.0, 0.0],
        [PITCH_LENGTH, 0.0],
        [PITCH_LENGTH, PITCH_WIDTH],
        [0.0, PITCH_WIDTH],
    ])

    H, H_inv = calibrate_homography(src_pts, dst_pts)
    if H is None:
        return None, None, False
    return H, H_inv, True

def get_pitch_contour_debug(frame):
    """Return an annotated frame showing detected pitch contour and corners."""
    if frame is None or frame.size == 0:
        return None, False
    src_pts, contour = _detect_pitch_corners(frame)
    if src_pts is None or contour is None:
        return frame.copy(), False

    dbg = frame.copy()
    cv2.drawContours(dbg, [contour], -1, (0, 0, 255), 3)
    for x, y in src_pts.astype(int):
        cv2.circle(dbg, (int(x), int(y)), 6, (255, 255, 0), -1)
    return dbg, True

def get_homography_hybrid(video_path=None):
    """
    Hybrid approach: try auto pitch contour first, then fallback to saved keypoints.
    Returns (H, H_inv).
    """
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            H, H_inv, ok = auto_homography_from_frame(frame)
            if ok:
                return H, H_inv

    return get_homography()

# Default pitch keypoints in metres (FIFA standard)
# These are the real-world coordinates of common pitch landmarks
DEFAULT_PITCH_POINTS = {
    "top_left_corner": (0, 0),
    "top_right_corner": (PITCH_LENGTH, 0),
    "bottom_left_corner": (0, PITCH_WIDTH),
    "bottom_right_corner": (PITCH_LENGTH, PITCH_WIDTH),
    "center_spot": (PITCH_LENGTH/2, PITCH_WIDTH/2),
    "left_penalty_spot": (11, PITCH_WIDTH/2),
    "right_penalty_spot": (PITCH_LENGTH-11, PITCH_WIDTH/2),
    "left_goal_center": (0, PITCH_WIDTH/2),
    "right_goal_center": (PITCH_LENGTH, PITCH_WIDTH/2),
    "top_left_penalty_box": (0, (PITCH_WIDTH-40.32)/2),
    "top_right_penalty_box": (16.5, (PITCH_WIDTH-40.32)/2),
    "bottom_left_penalty_box": (0, (PITCH_WIDTH+40.32)/2),
    "bottom_right_penalty_box": (16.5, (PITCH_WIDTH+40.32)/2),
}
