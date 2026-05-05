"""
Logs confirmed pick commands to stdout.

The actual robot communication happens via ServoPublisher (ZMQ port 4010),
which streams send_dict every frame to the robot's visual servoing process.
This module just records that a selection was confirmed.
"""

from __future__ import annotations

from typing import Optional


def send_command(
    target_point: tuple[int, int],
    sam_result: Optional[dict] = None,
) -> None:
    x, y = target_point

    if sam_result is None:
        print(f"[CMD] confirmed target=({x},{y})  — no SAM result yet, keep holding")
        return

    if "grasp_center_xyz" in sam_result:
        gx, gy, gz = sam_result["grasp_center_xyz"] * 100.0
        width_cm = sam_result["width_m"] * 100.0
        print(
            f"[CMD] confirmed target=({x},{y})  "
            f"grasp=({gx:.1f},{gy:.1f},{gz:.1f})cm  "
            f"width={width_cm:.1f}cm  → streaming to robot on port 4010"
        )
    else:
        cx, cy = sam_result["center_pix"]
        print(
            f"[CMD] confirmed target=({x},{y})  "
            f"mask_centre=({cx},{cy})  "
            f"(no depth — robot will not servo until depth available)"
        )
