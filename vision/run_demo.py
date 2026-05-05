"""
Vision pipeline demo — finger pointing selects an object via MobileSAM.

Run from the project root:
    python -m vision.run_demo
or:
    python vision/run_demo.py

Point at the robot camera image with your index finger. Hold the pointing
pose steady for CONFIRM_FRAMES frames to confirm the selection. MobileSAM
will segment the object under the cursor and start tracking it; the 3-D
grasp target is printed and can be forwarded to the visual servoing controller.

Press Q to quit.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from vision.image_source import ZMQImageSource
from vision.finger_pointer import FingerPointer, map_to_image_coords
from vision.command_sender import send_command
from vision.sam_tracker import SamTracker

WEBCAM_INDEX = 0
DISPLAY_HEIGHT = 480
CONFIRM_FRAMES = 60
STABILITY_RADIUS = 25  # pixels — cursor must stay within this circle before counting starts


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def _draw_cursor(img: np.ndarray, x: int, y: int, is_pointing: bool) -> np.ndarray:
    overlay = img.copy()
    cv2.circle(overlay, (x, y), 14, (0, 255, 0), 2)
    cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
    return cv2.addWeighted(overlay, 0.6, img, 0.4, 0)


def _draw_aruco_overlay(img: np.ndarray, markers: dict[int, tuple[int, int]]) -> np.ndarray:
    out = img.copy()
    for marker_id, (cx, cy) in markers.items():
        cv2.circle(out, (cx, cy), 14, (0, 255, 0), 2)
        cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.putText(out, str(marker_id), (cx + 16, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def _draw_cursor_label(img: np.ndarray, x: int, y: int, label: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, label, (x + 18, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _draw_confirm_bar(img: np.ndarray, held: int, total: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_w = int((held / total) * (w - 20))
    cv2.rectangle(out, (10, h - 50), (w - 10, h - 8), (60, 60, 60), -1)
    cv2.rectangle(out, (10, h - 50), (10 + bar_w, h - 8), (220, 100, 0), -1)
    return out


def _draw_sam_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros_like(img)
    overlay[mask > 0] = (0, 200, 255)
    out = cv2.addWeighted(img, 1.0, overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 200, 255), 2)
    return out


def main() -> None:
    image_source = ZMQImageSource(host="192.168.239.2", port=4405)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open webcam at index {WEBCAM_INDEX}.")
        sys.exit(1)

    pointer = FingerPointer()
    sam_tracker = SamTracker()

    held_frames = 0
    last_target = None
    needs_reset = False
    anchor_pos = None

    print("Vision demo running. Point at the robot image to place the cursor.")
    print("Hold a pointing pose to confirm and segment the object. Press Q to quit.")

    try:
        while True:
            ok, webcam_frame = cap.read()
            if not ok:
                break

            webcam_frame = cv2.flip(webcam_frame, 1)
            robot_frame = image_source.get_frame()

            result, annotated_webcam = pointer.process(webcam_frame)

            if robot_frame is None:
                left = _resize_to_height(annotated_webcam, DISPLAY_HEIGHT)
                placeholder = np.zeros((DISPLAY_HEIGHT, DISPLAY_HEIGHT, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for robot…", (20, DISPLAY_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                divider = np.full((DISPLAY_HEIGHT, 4, 3), 80, dtype=np.uint8)
                combined = np.concatenate([left, divider, placeholder], axis=1)
                cv2.imshow("Vision demo - press Q to quit", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            rh, rw = robot_frame.shape[:2]
            robot_display = robot_frame.copy()

            aruco_markers = image_source.get_aruco_markers() if hasattr(image_source, "get_aruco_markers") else {}
            if aruco_markers:
                robot_display = _draw_aruco_overlay(robot_display, aruco_markers)

            command_fired = False

            if result:
                px, py = map_to_image_coords(result.x_norm, result.y_norm, rw, rh)
                last_target = (px, py)

                if not result.is_pointing:
                    held_frames = 0
                    needs_reset = False
                    anchor_pos = None
                elif needs_reset:
                    held_frames = 0
                    anchor_pos = None
                else:
                    if anchor_pos is None:
                        anchor_pos = (px, py)
                    dx, dy = px - anchor_pos[0], py - anchor_pos[1]
                    if dx * dx + dy * dy > STABILITY_RADIUS ** 2:
                        held_frames = 0
                        anchor_pos = (px, py)
                    else:
                        held_frames += 1

                if held_frames >= CONFIRM_FRAMES:
                    sam_tracker.click(px, py)
                    command_fired = True
                    held_frames = 0
                    needs_reset = True

                robot_display = _draw_cursor(robot_display, px, py, result.is_pointing)

                if result.is_pointing and not needs_reset:
                    label = "hold still" if held_frames == 0 else "selecting..."
                    robot_display = _draw_cursor_label(robot_display, px, py, label)
                    robot_display = _draw_confirm_bar(robot_display, held_frames, CONFIRM_FRAMES)
            else:
                held_frames = 0
                needs_reset = False
                anchor_pos = None

            # SAM tracking — runs every frame; handles any click set above
            depth_frame = image_source.get_depth()
            camera_info = image_source.get_camera_info()
            depth_scale = image_source.get_depth_scale()
            sam_result = sam_tracker.process(
                robot_frame,
                depth_frame,
                camera_info,
                depth_scale if depth_scale is not None else 0.001,
            )

            if sam_result is not None:
                robot_display = _draw_sam_overlay(robot_display, sam_result["mask"])

            if command_fired:
                send_command(last_target, sam_result=sam_result)

            left = _resize_to_height(annotated_webcam, DISPLAY_HEIGHT)
            right = _resize_to_height(robot_display, DISPLAY_HEIGHT)

            divider = np.full((DISPLAY_HEIGHT, 4, 3), 80, dtype=np.uint8)
            combined = np.concatenate([left, divider, right], axis=1)

            cv2.putText(combined, "Webcam", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(combined, "Robot image", (left.shape[1] + 14, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow("Vision demo - press Q to quit", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        pointer.close()
        image_source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
