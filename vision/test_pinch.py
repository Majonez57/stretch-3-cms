"""
Local test for the pinch-to-select gesture — no robot required.

Run from the project root:
    python -m vision.test_pinch

Point your index finger to move the cursor.
Pinch (thumb + index tip together) to trigger a selection.
Press Q to quit.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from vision.finger_pointer import FingerPointer, map_to_image_coords

WEBCAM_INDEX = 0
WINDOW_NAME = "Pinch test - press Q to quit"
FLASH_FRAMES = 15


def _draw_cursor(img: np.ndarray, x: int, y: int, is_pinching: bool) -> np.ndarray:
    colour = (0, 255, 255) if is_pinching else (0, 255, 0)
    overlay = img.copy()
    cv2.circle(overlay, (x, y), 14, colour, 2)
    cv2.drawMarker(overlay, (x, y), colour, cv2.MARKER_CROSS, 20, 1)
    return cv2.addWeighted(overlay, 0.6, img, 0.4, 0)


def _draw_flash(img: np.ndarray, alpha: float) -> np.ndarray:
    overlay = np.full_like(img, (0, 255, 255))
    return cv2.addWeighted(img, 1.0 - alpha * 0.4, overlay, alpha * 0.4, 0)


def main() -> None:
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open webcam at index {WEBCAM_INDEX}.")
        sys.exit(1)

    pointer = FingerPointer()
    pinch_active = False
    flash_counter = 0
    select_count = 0

    print("Pinch test running. Point + pinch to select. Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result, annotated = pointer.process(frame)

            h, w = annotated.shape[:2]

            if result:
                px, py = map_to_image_coords(result.x_norm, result.y_norm, w, h)

                if result.is_pinching:
                    if not pinch_active:
                        select_count += 1
                        flash_counter = FLASH_FRAMES
                        pinch_active = True
                else:
                    pinch_active = False

                annotated = _draw_cursor(annotated, px, py, result.is_pinching)

                label = "pinch!" if result.is_pinching else ("pointing" if result.is_pointing else "")
                if label:
                    cv2.putText(annotated, label, (px + 18, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if result.is_pinching else (0, 255, 0),
                                1, cv2.LINE_AA)

            if flash_counter > 0:
                alpha = flash_counter / FLASH_FRAMES
                annotated = _draw_flash(annotated, alpha)
                cv2.putText(annotated, f"SELECTED  (x{select_count})",
                            (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, f"SELECTED  (x{select_count})",
                            (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
                flash_counter -= 1

            cv2.putText(annotated, "Point + pinch to select", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        pointer.close()
        cv2.destroyAllWindows()

    print(f"Done. Total selections: {select_count}")


if __name__ == "__main__":
    main()
