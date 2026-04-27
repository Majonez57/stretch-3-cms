"""
Vision pipeline demo — runs without SAM or a live robot.

Run from the project root:
    python -m vision.run_demo
or:
    python vision/run_demo.py

Press Q to quit.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from vision.image_source import PlaceholderImageSource
from vision.finger_pointer import FingerPointer, map_to_image_coords
from vision.command_sender import send_command

PLACEHOLDER_PATH = "assets/table_placeholder.jpg"
WEBCAM_INDEX = 0
DISPLAY_HEIGHT = 480
CONFIRM_FRAMES = 15


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def _draw_cursor(img: np.ndarray, x: int, y: int, is_pointing: bool) -> np.ndarray:
    out = img.copy()
    colour = (0, 120, 255) if is_pointing else (180, 180, 180)
    cv2.circle(out, (x, y), 12, colour, -1)
    cv2.circle(out, (x, y), 14, (255, 255, 255), 2)
    cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    return out


def _draw_confirm_bar(img: np.ndarray, held: int, total: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_w = int((held / total) * (w - 20))
    cv2.rectangle(out, (10, h - 20), (w - 10, h - 8), (60, 60, 60), -1)
    cv2.rectangle(out, (10, h - 20), (10 + bar_w, h - 8), (0, 200, 80), -1)
    return out


def main() -> None:
    try:
        image_source = PlaceholderImageSource(PLACEHOLDER_PATH)
    except FileNotFoundError:
        print(f"Placeholder image not found at '{PLACEHOLDER_PATH}'.")
        print("Add a photo of your table to assets/table_placeholder.jpg and rerun.")
        sys.exit(1)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open webcam at index {WEBCAM_INDEX}.")
        sys.exit(1)

    pointer = FingerPointer()
    held_frames = 0
    last_target = None

    print("Vision demo running. Point at the robot image to place the cursor.")
    print("Hold a pointing pose to confirm. Press Q to quit.")

    try:
        while True:
            ok, webcam_frame = cap.read()
            if not ok:
                break

            webcam_frame = cv2.flip(webcam_frame, 1)
            robot_frame = image_source.get_frame()

            result, annotated_webcam = pointer.process(webcam_frame)

            rh, rw = robot_frame.shape[:2]
            robot_display = robot_frame.copy()

            if result:
                px, py = map_to_image_coords(result.x_norm, result.y_norm, rw, rh)
                last_target = (px, py)

                if result.is_pointing:
                    held_frames += 1
                else:
                    held_frames = 0

                robot_display = _draw_cursor(robot_display, px, py, result.is_pointing)

                if result.is_pointing:
                    robot_display = _draw_confirm_bar(robot_display, held_frames, CONFIRM_FRAMES)

                if held_frames == CONFIRM_FRAMES:
                    send_command(last_target)
                    held_frames = 0
            else:
                held_frames = 0

            left = _resize_to_height(annotated_webcam, DISPLAY_HEIGHT)
            right = _resize_to_height(robot_display, DISPLAY_HEIGHT)

            divider = np.full((DISPLAY_HEIGHT, 4, 3), 80, dtype=np.uint8)
            combined = np.concatenate([left, divider, right], axis=1)

            cv2.putText(combined, "Webcam", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(combined, "Robot image", (left.shape[1] + 14, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow("Vision demo — press Q to quit", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        pointer.close()
        image_source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
