"""
Detects fingertip 3-D positions from the D405 wrist camera using the ArUco
markers mounted on the robot's gripper fingers.

Returns the fingertips dict expected by visual_servoing_demo.py:
    {
        'left':  {'pos': np.ndarray, 'x_axis': ..., 'y_axis': ..., 'z_axis': ...},
        'right': {'pos': np.ndarray, ...},
    }
An empty dict means neither finger marker was visible.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import yaml
from yaml.loader import SafeLoader

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_MBSAM_DIR = _REPO_ROOT / "mbsam"
_SERVO_DIR = pathlib.Path(__file__).parent.parent.parent / "stretch-3-cms"

for _p in [str(_MBSAM_DIR), str(_SERVO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aruco_detector as ad
import aruco_to_fingertips as af


class FingertipDetector:
    def __init__(
        self,
        marker_info_path: str | None = None,
        urdf_path: str | None = None,
    ) -> None:
        if marker_info_path is None:
            marker_info_path = str(_REPO_ROOT / "aruco_marker_info.yaml")

        with open(marker_info_path) as f:
            marker_info = yaml.load(f, Loader=SafeLoader)

        self._detector = ad.ArucoDetector(
            marker_info=marker_info,
            show_debug_images=False,
            use_apriltag_refinement=False,
            brighten_images=False,
        )

        if urdf_path is None:
            urdf_path = str(_SERVO_DIR / "stretch_uncalibrated.urdf")

        self._to_fingertips = af.ArucoToFingertips(
            urdf_filename=urdf_path,
            default_height_above_mounting_surface=af.suctioncup_height["cup_top"],
        )

    def detect(self, color_image: np.ndarray, camera_info: dict) -> dict:
        """Detect finger ArUco markers and return fingertips dict (may be empty)."""
        self._detector.update(color_image, camera_info)
        markers = self._detector.get_detected_marker_dict()
        return self._to_fingertips.get_fingertips(markers)
