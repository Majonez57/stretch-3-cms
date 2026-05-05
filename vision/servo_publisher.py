"""
ZMQ publisher that streams send_dict to the robot's visual servoing process.

The robot runs:
    python visual_servoing_demo.py -y -r

which subscribes a SUB socket to tcp://<dev_machine_ip>:4010 and reads
send_dict on every control loop iteration.

publish() must be called EVERY FRAME — the servo controller needs a
continuous stream of updated positions to track and grasp the object.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ServoPublisher:
    def __init__(self, port: int = 4010) -> None:
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required. Run: pip install pyzmq")

        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        address = f"tcp://*:{port}"
        self._socket.bind(address)
        print(f"[servo] Publishing on {address}  (robot connects SUB to this)")

    def publish(self, fingertips: dict, sam_result: Optional[dict]) -> None:
        """
        Build and send send_dict every frame.

        fingertips — from FingertipDetector.detect(); empty dict if not visible.
        sam_result — from SamTracker.process(); None until an object is selected.
        """
        yolo: list = []

        if sam_result is not None and "grasp_center_xyz" in sam_result:
            yolo = [{
                "name": "sam_tracked_object",
                "confidence": 1.0,
                "width_m": sam_result["width_m"],
                "estimated_z_m": sam_result["estimated_z_m"],
                "grasp_center_xyz": sam_result["grasp_center_xyz"],
                "left_side_xyz": sam_result.get("left_side_xyz", sam_result["grasp_center_xyz"]),
                "right_side_xyz": sam_result.get("right_side_xyz", sam_result["grasp_center_xyz"]),
            }]

        self._socket.send_pyobj({"fingertips": fingertips, "yolo": yolo})

    def close(self) -> None:
        self._socket.close()
