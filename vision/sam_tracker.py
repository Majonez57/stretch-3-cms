"""
SAM + Norfair object tracker for the vision pipeline.

Usage:
    tracker = SamTracker()
    tracker.click(px, py)          # call when finger confirmation fires
    result = tracker.process(      # call every frame
        color_frame, depth_frame, camera_info, depth_scale
    )

`result` is None until a click has been processed, then it's a dict with:
    "mask"             — HxW uint8 mask of the tracked object
    "center_pix"       — (cx, cy) pixel centre  (always present)
    "grasp_center_xyz" — 3-D grasp point in metres  (present when depth available)
    "width_m"          — estimated object width in metres
    "estimated_z_m"    — median depth of the object in metres
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class SamTracker:
    def __init__(self) -> None:
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError("ultralytics is required. Run: pip install ultralytics")
        try:
            from norfair import Tracker
        except ImportError:
            raise ImportError("norfair is required. Run: pip install norfair")

        print("[SAM] Loading MobileSAM model…")
        self._model = SAM("mobile_sam.pt")
        self._Tracker = Tracker
        self._tracker = Tracker(distance_function="euclidean", distance_threshold=80)
        self._clicked_point: Optional[tuple[int, int]] = None
        self._current_mask: Optional[np.ndarray] = None
        self._last_depth_result: Optional[dict] = None
        self._frames_since_resegment: int = 0
        print("[SAM] Ready.")

    def click(self, x: int, y: int) -> None:
        self._clicked_point = (x, y)
        self._last_depth_result = None
        self._frames_since_resegment = 0

    def reset(self) -> None:
        self._clicked_point = None
        self._current_mask = None
        self._last_depth_result = None
        self._frames_since_resegment = 0
        self._tracker = self._Tracker(distance_function="euclidean", distance_threshold=80)

    def process(
        self,
        color_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        camera_info: Optional[dict] = None,
        depth_scale: float = 0.001,
        neg_points: Optional[list] = None,
    ) -> Optional[dict]:
        from norfair import Detection

        if self._clicked_point is not None:
            self._current_mask = self._segment_from_point(color_frame, self._clicked_point, neg_points)
            self._clicked_point = None

        if self._current_mask is None:
            self._tracker.update(detections=[])
            return None

        centroid = self._mask_centroid(self._current_mask)
        detections = [Detection(points=centroid)] if centroid is not None else []
        tracked_objects = self._tracker.update(detections=detections)

        self._frames_since_resegment += 1
        if self._frames_since_resegment >= 5:
            self._frames_since_resegment = 0
            for obj in tracked_objects:
                cx, cy = obj.estimate[0].astype(int)
                refreshed = self._segment_from_point(color_frame, (cx, cy), neg_points)
                if refreshed is not None:
                    self._current_mask = refreshed

        if self._current_mask is None:
            return None

        box = self._mask_to_box(self._current_mask)
        if box is None:
            return None

        min_x, min_y, max_x, max_y = box
        cx_pix = int((min_x + max_x) / 2)
        cy_pix = int((min_y + max_y) / 2)

        result: dict = {"mask": self._current_mask, "center_pix": (cx_pix, cy_pix)}

        if depth_frame is not None and camera_info is not None:
            depth_crop = depth_frame[min_y:max_y, min_x:max_x]
            mask_crop = self._current_mask[min_y:max_y, min_x:max_x]
            valid_depths = depth_crop[mask_crop > 0]
            valid_depths = valid_depths[valid_depths > 0]

            if valid_depths.size > 0:
                estimated_z_m = float(np.percentile(valid_depths, 50)) * depth_scale

                center_xyz = self._pixel_to_3d([cx_pix, cy_pix], estimated_z_m, camera_info)
                left_xyz = self._pixel_to_3d([min_x, cy_pix], estimated_z_m, camera_info)
                right_xyz = self._pixel_to_3d([max_x, cy_pix], estimated_z_m, camera_info)
                width_m = float(np.linalg.norm(right_xyz - left_xyz))

                center_ray = center_xyz / np.linalg.norm(center_xyz)
                grasp_center_xyz = center_xyz + (width_m / 2.0) * center_ray

                self._last_depth_result = {
                    "grasp_center_xyz": grasp_center_xyz,
                    "left_side_xyz": left_xyz,
                    "right_side_xyz": right_xyz,
                    "width_m": width_m,
                    "estimated_z_m": estimated_z_m,
                }

        if self._last_depth_result is not None:
            result.update(self._last_depth_result)

        return result

    # ------------------------------------------------------------------

    def _segment_from_point(
        self, frame: np.ndarray, point: tuple[int, int],
        neg_points: Optional[list] = None,
    ) -> Optional[np.ndarray]:
        x, y = point
        all_points = [[x, y]]
        all_labels = [1]
        if neg_points:
            for nx, ny in neg_points:
                all_points.append([int(nx), int(ny)])
                all_labels.append(0)
        results = self._model.predict(
            source=frame, points=[all_points], labels=[all_labels], verbose=False
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
