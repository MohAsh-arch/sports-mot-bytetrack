"""
─────────────────────────────────────────────
src/detection.py — Player Detection
─────────────────────────────────────────────
Loads YOLOv8m, wraps with SAHI for panoramic
mode, returns sv.Detections.
Consolidates Cell 3 run_detection().
─────────────────────────────────────────────
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, PLAYER_MODEL_NAME, PERSON_CLASS_ID,
    DET_CONF, DET_IOU,
    SAHI_SLICE_H, SAHI_SLICE_W, SAHI_OVERLAP,
    PANORAMIC_RATIO,
)


class PlayerDetector:
    """
    Detects persons in a frame using YOLOv8m.
    Automatically uses SAHI slicing for panoramic
    (wide) footage where players are small.
    """

    def __init__(self, is_panoramic: bool = False,
                 device: str = DEVICE,
                 det_conf: float = DET_CONF,
                 det_iou: float = DET_IOU):
        self.device = device
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.is_panoramic = is_panoramic

        # Load YOLO model
        print(f"Loading {PLAYER_MODEL_NAME} on {device} ...")
        self.model = YOLO(PLAYER_MODEL_NAME)
        self.model.to(device)

        # Load SAHI wrapper if panoramic
        self.sahi_model = None
        if is_panoramic:
            from sahi import AutoDetectionModel
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=PLAYER_MODEL_NAME,
                confidence_threshold=det_conf,
                device=device,
            )
            print(f"  SAHI enabled: slice {SAHI_SLICE_W}x{SAHI_SLICE_H}, "
                  f"overlap {SAHI_OVERLAP}")

        print(f"  Detection: conf={det_conf}, iou={det_iou}")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a single frame.
        Returns sv.Detections with person class only.
        """
        if self.sahi_model is not None:
            return self._detect_sahi(frame)
        else:
            return self._detect_direct(frame)

    def _detect_direct(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(
            frame,
            conf=self.det_conf,
            iou=self.det_iou,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)

    def _detect_sahi(self, frame: np.ndarray) -> sv.Detections:
        from sahi.predict import get_sliced_prediction

        sahi_result = get_sliced_prediction(
            frame,
            self.sahi_model,
            slice_height=SAHI_SLICE_H,
            slice_width=SAHI_SLICE_W,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            verbose=0,
        )

        boxes, confs, class_ids = [], [], []
        for obj in sahi_result.object_prediction_list:
            if obj.score.value < self.det_conf:
                continue
            if obj.category.name.lower() != "person":
                continue
            b = obj.bbox
            boxes.append([b.minx, b.miny, b.maxx, b.maxy])
            confs.append(obj.score.value)
            class_ids.append(obj.category.id)

        if len(boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )


def detect_video_mode(video_path: str) -> tuple:
    """
    Auto-detect video parameters and whether it's panoramic.
    Returns (fps, width, height, total_frames, is_panoramic).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    aspect = w / h if h > 0 else 1.0
    is_panoramic = aspect > PANORAMIC_RATIO

    return fps, w, h, total, is_panoramic
