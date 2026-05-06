"""
Fingertip tracking for point-and-segment interaction.

FingerPointer tracks the index fingertip position in a webcam frame using
MediaPipe Hands and detects when the user is in a pointing pose (index
extended, remaining fingers curled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# MediaPipe landmark indices used for pointing detection
_THUMB_TIP = 4
_INDEX_TIP = 8
_INDEX_PIP = 6
_MIDDLE_TIP = 12
_MIDDLE_PIP = 10
_RING_TIP = 16
_RING_PIP = 14
_PINKY_TIP = 20
_PINKY_PIP = 18

_PINCH_THRESHOLD = 0.07  # normalised distance between thumb and index tips

# Drawing colours (BGR)
_COLOUR_POINTING = (0, 120, 255)
_COLOUR_PINCH = (0, 255, 255)
_COLOUR_IDLE = (180, 180, 180)
_COLOUR_TIP = (0, 0, 255)

# MediaPipe hand connections for skeleton drawing
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

DEFAULT_MODEL_PATH = "gesture/hand_landmarker.task"


@dataclass
class PointerResult:
    x_norm: float
    y_norm: float
    is_pointing: bool
    is_pinching: bool = False


class FingerPointer:

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is not installed. Run: pip install mediapipe")

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process(
        self, frame: np.ndarray
    ) -> Tuple[Optional[PointerResult], np.ndarray]:
        annotated = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, annotated

        lm = result.hand_landmarks[0]
        tip = lm[_INDEX_TIP]
        is_pointing = _detect_pointing(lm)
        is_pinching = _detect_pinch(lm)

        _draw_hand(annotated, lm, is_pointing)
        _draw_fingertip(annotated, tip.x, tip.y, is_pointing, is_pinching)

        return PointerResult(x_norm=tip.x, y_norm=tip.y, is_pointing=is_pointing, is_pinching=is_pinching), annotated

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FingerPointer":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_to_image_coords(
    x_norm: float, y_norm: float, image_width: int, image_height: int
) -> Tuple[int, int]:
    x = int(np.clip(x_norm, 0.0, 1.0) * image_width)
    y = int(np.clip(y_norm, 0.0, 1.0) * image_height)
    return x, y


def _detect_pointing(lm) -> bool:
    index_extended = lm[_INDEX_TIP].y < lm[_INDEX_PIP].y
    middle_curled = lm[_MIDDLE_TIP].y > lm[_MIDDLE_PIP].y
    ring_curled = lm[_RING_TIP].y > lm[_RING_PIP].y
    pinky_curled = lm[_PINKY_TIP].y > lm[_PINKY_PIP].y
    return index_extended and middle_curled and ring_curled and pinky_curled


def _detect_pinch(lm) -> bool:
    tx, ty = lm[_THUMB_TIP].x, lm[_THUMB_TIP].y
    ix, iy = lm[_INDEX_TIP].x, lm[_INDEX_TIP].y
    dist = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
    return dist < _PINCH_THRESHOLD


def _draw_hand(frame: np.ndarray, lm: list, is_pointing: bool) -> None:
    h, w = frame.shape[:2]
    colour = _COLOUR_POINTING if is_pointing else _COLOUR_IDLE

    for a, b in _HAND_CONNECTIONS:
        x1, y1 = int(lm[a].x * w), int(lm[a].y * h)
        x2, y2 = int(lm[b].x * w), int(lm[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), colour, 2)

    for landmark in lm:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 4, colour, -1)


def _draw_fingertip(
    frame: np.ndarray, x_norm: float, y_norm: float, is_pointing: bool, is_pinching: bool = False
) -> None:
    h, w = frame.shape[:2]
    cx = int(x_norm * w)
    cy = int(y_norm * h)
    if is_pinching:
        colour = _COLOUR_PINCH
    elif is_pointing:
        colour = _COLOUR_TIP
    else:
        colour = _COLOUR_IDLE
    cv2.circle(frame, (cx, cy), 10, colour, -1)
    cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
