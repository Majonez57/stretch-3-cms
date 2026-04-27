"""
Command sender stub.

send_command() is the integration point between the vision pipeline and the
robot. For now it just prints what it would do. When SAM is added it will
receive a mask; when the robot integration is wired up it will call the
ROS 2 bridge or ZMQ visual servoing controller.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def send_command(
    target_point: tuple[int, int],
    mask: Optional[np.ndarray] = None,
) -> None:
    """
    Send a pick/interact command targeting the pointed-at object.

    Args:
        target_point: (x, y) pixel coords on the robot image.
        mask:         Boolean HxW mask from SAM (None until SAM is wired in).
    """
    x, y = target_point

    if mask is not None:
        ys, xs = np.where(mask)
        cx = int(xs.mean()) if len(xs) else x
        cy = int(ys.mean()) if len(ys) else y
        print(f"[CMD] target=({x}, {y})  mask_centroid=({cx}, {cy})  mask_px={mask.sum()}")
    else:
        print(f"[CMD] target=({x}, {y})  mask=None (SAM not yet wired)")
