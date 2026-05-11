"""
src/ball.py — Ball Detection + Kalman Gate Filter
FIX #2: Dedicated ball model + ghost dot rejection
"""

import cv2, time, urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter as KF
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, BALL_MODEL_URL, BALL_MODEL_URL_FALLBACKS, BALL_MODEL_PATH,
    BALL_CONF, BALL_CONF_DEDICATED, BALL_IOU,
    BALL_SAHI_SLICE, BALL_SAHI_OVERLAP, BALL_COCO_CLASS_ID, MAX_BALL_JUMP,
)

class BallKalmanGate:
    """
    Kalman filter gate for ball trajectory.
    Rejects detections that are > MAX_BALL_JUMP pixels
    from the predicted position (ghost dot filter).
    """
    def __init__(self, max_jump=MAX_BALL_JUMP):
        self.kf = KF(dim_x=4, dim_z=2)  # [x, y, vx, vy]
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.kf.R *= 50   # measurement noise
        self.kf.P *= 500  # initial covariance
        self.max_jump = max_jump
        self.initialized = False

    def update(self, cx, cy):
        """Returns True if accepted, False if ghost rejected."""
        if not self.initialized:
            # Seed the filter with the first detection (matches notebook Cell 5)
            self.kf.x = np.array([[cx],[cy],[0],[0]], dtype=float)
            self.initialized = True
            self.kf.update(np.array([[cx],[cy]], dtype=float))
            return True
        # Predict THEN gate-check (notebook order: predict → distance → accept/reject)
        self.kf.predict()
        pred = self.kf.x[:2].flatten()
        dist = np.linalg.norm(np.array([cx, cy]) - pred)
        if dist < self.max_jump:
            self.kf.update(np.array([[cx],[cy]], dtype=float))
            return True
        return False  # Ghost dot rejected

    def get_predicted(self):
        """Get current predicted position without advancing the filter state.
        Uses a copy to avoid double-predict drift on skipped frames."""
        if not self.initialized:
            return None
        # Return prediction without mutating internal state
        # (avoid calling self.kf.predict() which would advance state)
        F = self.kf.F
        predicted_x = F @ self.kf.x
        return predicted_x[:2].flatten()

class BallDetector:
    """
    Ball detection with dedicated model + Kalman gate.
    Falls back to COCO sports ball class if dedicated model unavailable.
    """
    def __init__(self, device=DEVICE, use_sahi=True):
        self.device = device
        self.use_sahi = use_sahi
        self.dedicated = False
        self.gate = BallKalmanGate()
        self.ghost_rejected = 0

        # Try dedicated ball model first
        if self._ensure_ball_model():
            self.model = YOLO(str(BALL_MODEL_PATH))
            self.model.to(device)
            self.dedicated = True
            self.ball_class_id = 0  # dedicated model: class 0 = ball
            self.ball_conf = BALL_CONF_DEDICATED
            print(f"Ball detector: dedicated model loaded")
        else:
            # Fallback: COCO sports ball
            self.model = YOLO("yolov8x.pt")
            self.model.to(device)
            self.ball_class_id = BALL_COCO_CLASS_ID
            self.ball_conf = BALL_CONF
            print(f"Ball detector: COCO fallback (class {BALL_COCO_CLASS_ID})")

        # SAHI wrapper
        self.sahi_model = None
        if use_sahi:
            try:
                from sahi import AutoDetectionModel
                model_path = str(BALL_MODEL_PATH) if self.dedicated else "yolov8x.pt"
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics", model_path=model_path,
                    confidence_threshold=self.ball_conf, device=device,
                )
            except Exception as e:
                print(f"SAHI for ball not available: {e}")

        print(f"  Kalman gate: max_jump={MAX_BALL_JUMP}px")

    def _ensure_ball_model(self):
        if BALL_MODEL_PATH.exists():
            return True
        # Try primary URL, then fallbacks
        urls_to_try = [BALL_MODEL_URL] + BALL_MODEL_URL_FALLBACKS
        BALL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        for url in urls_to_try:
            try:
                print(f"Downloading dedicated ball detection model from {url}...")
                urllib.request.urlretrieve(url, str(BALL_MODEL_PATH))
                # Verify the file is a valid model (not an HTML error page)
                if BALL_MODEL_PATH.stat().st_size > 1_000_000:  # >1MB = likely valid
                    print(f"  Downloaded → {BALL_MODEL_PATH}")
                    return True
                else:
                    print(f"  Downloaded file too small ({BALL_MODEL_PATH.stat().st_size} bytes), trying next URL")
                    BALL_MODEL_PATH.unlink(missing_ok=True)
            except Exception as e:
                print(f"  Failed from {url}: {e}")
                BALL_MODEL_PATH.unlink(missing_ok=True)
        print("Could not download ball model from any URL — using COCO fallback")
        return False

    def detect_frame(self, frame):
        """
        Detect ball in a single frame.
        Returns (cx, cy, conf) or None.
        Applies Kalman gate to reject ghost dots.
        """
        raw = self._raw_detect(frame)
        if raw is None:
            return None

        cx, cy, conf = raw
        if self.gate.update(cx, cy):
            return (cx, cy, conf)
        else:
            self.ghost_rejected += 1
            return None  # Ghost dot rejected

    def _raw_detect(self, frame):
        """Raw detection without Kalman gate."""
        if self.sahi_model is not None:
            return self._detect_sahi(frame)
        return self._detect_direct(frame)

    def _detect_direct(self, frame):
        classes = [self.ball_class_id] if not self.dedicated else None
        results = self.model(frame, conf=self.ball_conf, iou=BALL_IOU,
                            classes=classes, verbose=False)[0]
        if len(results.boxes) == 0:
            return None
        # Filter to ball class
        boxes = results.boxes
        if not self.dedicated:
            mask = boxes.cls.cpu().numpy().astype(int) == self.ball_class_id
            if not mask.any():
                return None
            idx = np.where(mask)[0]
            confs = boxes.conf.cpu().numpy()[idx]
            best_i = idx[np.argmax(confs)]
        else:
            best_i = boxes.conf.cpu().numpy().argmax()
        bx = boxes.xyxy[best_i].cpu().numpy()
        conf_val = float(boxes.conf[best_i].cpu().numpy())
        cx = (bx[0]+bx[2])/2; cy = (bx[1]+bx[3])/2
        return (round(float(cx),1), round(float(cy),1), round(conf_val,3))

    def _detect_sahi(self, frame):
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            frame, self.sahi_model,
            slice_height=BALL_SAHI_SLICE, slice_width=BALL_SAHI_SLICE,
            overlap_height_ratio=BALL_SAHI_OVERLAP,
            overlap_width_ratio=BALL_SAHI_OVERLAP, verbose=0,
        )
        preds = [o for o in result.object_prediction_list
                 if o.score.value >= self.ball_conf]
        if not self.dedicated:
            preds = [o for o in preds if o.category.id == self.ball_class_id]
        if not preds:
            return None
        best = max(preds, key=lambda o: o.score.value)
        b = best.bbox
        cx = (b.minx+b.maxx)/2; cy = (b.miny+b.maxy)/2
        return (round(cx,1), round(cy,1), round(best.score.value,3))

def run_ball_detection(video_path, frames_to_run, device=DEVICE,
                       is_panoramic=True, progress_cb=None):
    """Run ball detection on video. Returns ball_df."""
    detector = BallDetector(device=device, use_sahi=is_panoramic)
    cap = cv2.VideoCapture(str(video_path))
    rows = []; idx = 0; detected = 0; t0 = time.time()
    while idx < frames_to_run:
        ret, frame = cap.read()
        if not ret: break
        result = detector.detect_frame(frame)
        if result:
            cx,cy,conf = result; detected += 1
            rows.append({"frame":idx,"cx":cx,"cy":cy,"conf":conf})
        else:
            rows.append({"frame":idx,"cx":float("nan"),
                        "cy":float("nan"),"conf":float("nan")})
        if progress_cb and idx % 10 == 0:
            progress_cb(idx, frames_to_run, time.time()-t0)
        idx += 1
    cap.release()
    df = pd.DataFrame(rows)
    pct = detected/idx*100 if idx > 0 else 0
    print(f"Ball: {detected}/{idx} frames ({pct:.1f}%), "
          f"{detector.ghost_rejected} ghosts rejected")
    return df
