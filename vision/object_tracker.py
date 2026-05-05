"""
CSRT + MobileSAM object tracker for the vision pipeline.

Drop-in replacement for SamTracker with the same interface.

Strategy:
  - Click: SAM segments the object, CSRT tracker is initialised from the mask bbox.
  - Every frame: CSRT updates the bounding box (fast, handles scale as the
    camera approaches the object).
  - Every SAM_REFRESH_FRAMES: SAM re-runs from the CSRT centre to keep the
    mask accurate and re-initialise CSRT if the box has drifted.
  - Once a valid grasp_center_xyz is captured the mask and 3-D position are
    frozen so the gripper cannot corrupt the measurement.

Usage:
    tracker = ObjectTracker()
    tracker.click(px, py)
    result = tracker.process(color_frame, depth_frame, camera_info, depth_scale)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

SAM_REFRESH_FRAMES = 10


class ObjectTracker:
    def __init__(self) -> None:
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError("ultralytics is required. Run: pip install ultralytics")

        print("[Tracker] Loading MobileSAM model…")
        self._model = SAM("mobile_sam.pt")
        self._clicked_point: Optional[tuple[int, int]] = None
        self._current_mask: Optional[np.ndarray] = None
        self._cv_tracker: Optional[cv2.Tracker] = None
        self._box: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
        self._last_depth_result: Optional[dict] = None
        self._frames_since_sam: int = 0
        self._grasp_locked: bool = False
        print("[Tracker] Ready.")

    def click(self, x: int, y: int) -> None:
        self._clicked_point = (x, y)
        self._last_depth_result = None
        self._frames_since_sam = 0
        self._grasp_locked = False
        self._cv_tracker = None
        self._box = None
        self._current_mask = None

    def reset(self) -> None:
        self._clicked_point = None
        self._current_mask = None
        self._cv_tracker = None
        self._box = None
        self._last_depth_result = None
        self._frames_since_sam = 0
        self._grasp_locked = False

    def process(
        self,
        color_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        camera_info: Optional[dict] = None,
        depth_scale: float = 0.001,
    ) -> Optional[dict]:

        # --- Initial SAM segmentation on click ---
        if self._clicked_point is not None:
            mask = self._segment_from_point(color_frame, self._clicked_point)
            self._clicked_point = None
            if mask is not None:
                self._current_mask = mask
                self._cv_tracker, self._box = self._init_csrt(color_frame, mask)
            self._frames_since_sam = 0

        if self._current_mask is None:
            return None

        # --- Per-frame CSRT update ---
        if not self._grasp_locked and self._cv_tracker is not None:
            ok, box = self._cv_tracker.update(color_frame)
            if ok:
                self._box = tuple(int(v) for v in box)
            else:
                # Tracker lost — fall back to mask centroid and re-init
                self._cv_tracker = None

        # --- Periodic SAM refresh to correct drift ---
        if not self._grasp_locked:
            self._frames_since_sam += 1
            if self._frames_since_sam >= SAM_REFRESH_FRAMES:
                self._frames_since_sam = 0
                cx, cy = self._box_centre()
                if cx is not None:
                    refreshed = self._segment_from_point(color_frame, (cx, cy))
                    if refreshed is not None:
                        self._current_mask = refreshed
                        self._cv_tracker, self._box = self._init_csrt(color_frame, refreshed)

        # --- Derive bounding coords ---
        if self._box is not None:
            x, y, w, h = self._box
            min_x, min_y, max_x, max_y = x, y, x + w, y + h
        else:
            coords = self._mask_to_box(self._current_mask)
            if coords is None:
                return None
            min_x, min_y, max_x, max_y = coords

        cx_pix = (min_x + max_x) // 2
        cy_pix = (min_y + max_y) // 2
        result: dict = {"mask": self._current_mask, "center_pix": (cx_pix, cy_pix)}

        # --- Depth estimation (skipped once locked) ---
        if not self._grasp_locked and depth_frame is not None and camera_info is not None:
            fh, fw = depth_frame.shape[:2]
            bx0, by0 = max(min_x, 0), max(min_y, 0)
            bx1, by1 = min(max_x, fw), min(max_y, fh)

            depth_crop = depth_frame[by0:by1, bx0:bx1]
            mask_crop = self._current_mask[by0:by1, bx0:bx1]
            valid = depth_crop[mask_crop > 0]
            valid = valid[valid > 0]

            if valid.size > 0:
                z = float(np.percentile(valid, 50)) * depth_scale
                center_xyz = self._pixel_to_3d([cx_pix, cy_pix], z, camera_info)
                left_xyz   = self._pixel_to_3d([min_x, cy_pix], z, camera_info)
                right_xyz  = self._pixel_to_3d([max_x, cy_pix], z, camera_info)
                width_m    = float(np.linalg.norm(right_xyz - left_xyz))
                ray        = center_xyz / np.linalg.norm(center_xyz)

                self._last_depth_result = {
                    "grasp_center_xyz": center_xyz + (width_m / 2.0) * ray,
                    "left_side_xyz":    left_xyz,
                    "right_side_xyz":   right_xyz,
                    "width_m":          width_m,
                    "estimated_z_m":    z,
                }
                self._grasp_locked = True

        if self._last_depth_result is not None:
            result.update(self._last_depth_result)

        return result

    # ------------------------------------------------------------------

    def _init_csrt(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> tuple[Optional[cv2.Tracker], Optional[tuple]]:
        coords = self._mask_to_box(mask)
        if coords is None:
            return None, None
        min_x, min_y, max_x, max_y = coords
        xywh = (min_x, min_y, max_x - min_x, max_y - min_y)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, xywh)
        return tracker, xywh

    def _box_centre(self) -> tuple[Optional[int], Optional[int]]:
        if self._box is not None:
            x, y, w, h = self._box
            return x + w // 2, y + h // 2
        centroid = self._mask_centroid(self._current_mask)
        if centroid is not None:
            return int(centroid[0]), int(centroid[1])
        return None, None

    def _segment_from_point(
        self, frame: np.ndarray, point: tuple[int, int]
    ) -> Optional[np.ndarray]:
        x, y = point
        results = self._model.predict(
            source=frame, points=[[x, y]], labels=[1], verbose=False
        )
        if not results or results[0].masks is None:
            return None
        mask = results[0].masks.data[0].cpu().numpy()
        return cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return np.array([xs.mean(), ys.mean()])

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _pixel_to_3d(uv: list[float], depth_m: float, camera_info: dict) -> np.ndarray:
        K = camera_info["camera_matrix"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        return np.array([
            (uv[0] - cx) * depth_m / fx,
            (uv[1] - cy) * depth_m / fy,
            depth_m,
        ])
