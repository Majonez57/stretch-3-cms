"""
Command sender stub.

send_command() is the integration point between the vision pipeline and the
robot. For now it just prints what it would do. Wire up the ZMQ / ROS 2 call
here when the visual servoing controller is ready.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def send_command(
    target_point: tuple[int, int],
    sam_result: Optional[dict] = None,
) -> None:
    """
    Send a pick/interact command for the pointed-at object.

    Args:
        target_point: (x, y) pixel coords on the robot image where the user pointed.
        sam_result:   Dict returned by SamTracker.process(), or None if SAM has no result yet.
                      Contains at minimum "mask" and "center_pix".
                      Contains "grasp_center_xyz", "width_m", "estimated_z_m" when depth is available.
    """
    x, y = target_point

    if sam_result is None:
        print(f"[CMD] target=({x},{y})  sam=None (no object segmented yet)")
        return

    if "grasp_center_xyz" in sam_result:
        gx, gy, gz = sam_result["grasp_center_xyz"] * 100.0
        width_cm = sam_result["width_m"] * 100.0
        print(
            f"[CMD] target=({x},{y})  "
            f"grasp=({gx:.1f},{gy:.1f},{gz:.1f})cm  "
            f"width={width_cm:.1f}cm"
        )
    else:
        mask = sam_result["mask"]
        cx, cy = sam_result["center_pix"]
        print(
            f"[CMD] target=({x},{y})  "
            f"mask_centre=({cx},{cy})  "
            f"mask_px={int(mask.sum())}"
        )
