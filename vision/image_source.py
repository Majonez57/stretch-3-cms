"""
Image source abstraction for the robot camera feed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class ImageSource(ABC):
    """Common interface for all robot image sources."""

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame, or None if not yet available."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class PlaceholderImageSource(ImageSource):
    """
    Serves a single static image as if it were a live camera feed.
    """

    def __init__(self, image_path: str) -> None:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load placeholder image: {image_path}")
        self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        return self._frame.copy()

    def close(self) -> None:
        pass


class ZMQImageSource(ImageSource):
    """
    Receives BGR frames from the robot's D405 camera over ZMQ.

    The robot side runs robot/send_camera_images.py, which publishes a dict:
        {
            'color_image':       np.ndarray (BGR, HxWx3),
            'depth_image':       np.ndarray (uint16, HxW),
            'color_camera_info': dict,
            'depth_camera_info': dict,
            'depth_scale':       float,
        }
    """

    def __init__(self, host: str, port: int) -> None:
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required for ZMQImageSource. Run: pip install pyzmq")

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._camera_info: Optional[dict] = None
        self._aruco_markers: dict = {}

    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest color frame. Non-blocking — returns cached frame if no new data."""
        try:
            data = self._socket.recv_pyobj(flags=self._zmq.NOBLOCK)
            self._latest_frame = data.get("color_image")
            self._latest_depth = data.get("depth_image")
            self._camera_info = data.get("color_camera_info")
            self._aruco_markers = data.get("aruco_markers", {})
        except self._zmq.Again:
            pass
        return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_depth(self) -> Optional[np.ndarray]:
        """Return latest depth frame (uint16). Call get_frame() first to refresh."""
        return self._latest_depth.copy() if self._latest_depth is not None else None

    def get_camera_info(self) -> Optional[dict]:
        """Return camera intrinsics dict from the last received frame."""
        return self._camera_info

    def get_aruco_markers(self) -> dict[int, tuple[int, int]]:
        """Return {marker_id: (cx, cy)} for ArUco markers visible in the last frame."""
        return self._aruco_markers

    def close(self) -> None:
        self._socket.close()
        self._context.term()
