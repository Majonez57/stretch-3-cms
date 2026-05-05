"""
Vision pipeline — finger pointing selects an object via MobileSAM, then
streams its 3-D position to the robot's visual servoing controller.

Run from the project root:
    python -m vision.run_demo

Point at the robot camera image with your index finger and hold the pointing
pose steady for CONFIRM_FRAMES to confirm the target. MobileSAM segments the
object and starts tracking it. The 3-D grasp position and fingertip data are
published on ZMQ port 4010 every frame so the robot can servo to the object.

Robot side — run on the Stretch:
    python robot/send_camera_images.py          # D405 stream → port 4405
    python visual_servoing_demo.py -y -r        # servo controller (yolo+remote)

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
from vision.fingertip_detector import FingertipDetector
from vision.servo_publisher import ServoPublisher

WEBCAM_INDEX = 0
DISPLAY_HEIGHT = 480
CONFIRM_FRAMES = 60
STABILITY_RADIUS = 25
WINDOW_NAME = "Vision demo - press Q to quit"
USE_FINGER_POINTER = False  # set True to re-enable MediaPipe webcam pointing

# Shared state for mouse click handler
_click_state: dict = {"pending": None, "left_w": 0, "robot_scale": 1.0}


def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        robot_x_in_panel = x - (_click_state["left_w"] + 4)  # 4px divider
        if robot_x_in_panel >= 0:
            scale = _click_state["robot_scale"]
            _click_state["pending"] = (int(robot_x_in_panel / scale), int(y / scale))


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

    cap = cv2.VideoCapture(WEBCAM_INDEX) if USE_FINGER_POINTER else None
    if USE_FINGER_POINTER and not cap.isOpened():
        print(f"Could not open webcam at index {WEBCAM_INDEX}.")
        sys.exit(1)

    pointer = FingerPointer() if USE_FINGER_POINTER else None
    sam_tracker = SamTracker()
    fingertip_detector = FingertipDetector()
    servo_publisher = ServoPublisher(port=4010)

    held_frames = 0
    last_target = None
    needs_reset = False
    anchor_pos = None

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    print("Vision pipeline running.")
    if USE_FINGER_POINTER:
        print("Point at the robot image and hold to select an object.")
    print("Left-click on the robot image to select an object.")
    print("send_dict streamed to robot on port 4010 every frame.")
    print("Press Q to quit.")

    try:
        while True:
            if USE_FINGER_POINTER:
                ok, webcam_frame = cap.read()
                if not ok:
                    break
                webcam_frame = cv2.flip(webcam_frame, 1)
                result, annotated_webcam = pointer.process(webcam_frame)
            else:
                result = None
                annotated_webcam = None

            robot_frame = image_source.get_frame()

            if robot_frame is None:
                servo_publisher.publish({}, None)
                placeholder = np.zeros((DISPLAY_HEIGHT, DISPLAY_HEIGHT, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for robot...", (20, DISPLAY_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                if annotated_webcam is not None:
                    left = _resize_to_height(annotated_webcam, DISPLAY_HEIGHT)
                    divider = np.full((DISPLAY_HEIGHT, 4, 3), 80, dtype=np.uint8)
                    cv2.imshow(WINDOW_NAME, np.concatenate([left, divider, placeholder], axis=1))
                else:
                    cv2.imshow(WINDOW_NAME, placeholder)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            depth_frame = image_source.get_depth()
            depth_camera_info = image_source.get_depth_camera_info()
            camera_info = image_source.get_camera_info()
            depth_scale = image_source.get_depth_scale()

            # Use depth camera info for accuracy; fall back to colour
            active_camera_info = depth_camera_info or camera_info

            # --- Fingertip detection ---
            fingertips: dict = {}
            if active_camera_info is not None:
                try:
                    fingertips = fingertip_detector.detect(robot_frame, active_camera_info)
                except Exception:
                    pass

            rh, rw = robot_frame.shape[:2]
            robot_display = robot_frame.copy()

            aruco_markers = image_source.get_aruco_markers() if hasattr(image_source, "get_aruco_markers") else {}
            if aruco_markers:
                robot_display = _draw_aruco_overlay(robot_display, aruco_markers)

            # --- Finger pointer → confirmation ---
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

            # --- Mouse click (instant select, bypasses finger hold) ---
            if _click_state["pending"] is not None:
                click_pt = _click_state["pending"]
                sam_tracker.click(*click_pt)
                last_target = click_pt
                _click_state["pending"] = None
                command_fired = True

            # --- SAM tracking ---
            sam_result = sam_tracker.process(
                robot_frame,
                depth_frame,
                active_camera_info,
                depth_scale if depth_scale is not None else 0.001,
            )

            if sam_result is not None:
                robot_display = _draw_sam_overlay(robot_display, sam_result["mask"])

            # --- Publish every frame to visual servoing ---
            servo_publisher.publish(fingertips, sam_result)

            if command_fired:
                send_command(last_target, sam_result=sam_result)

            # --- Display ---
            right = _resize_to_height(robot_display, DISPLAY_HEIGHT)
            _click_state["robot_scale"] = DISPLAY_HEIGHT / rh

            if USE_FINGER_POINTER and annotated_webcam is not None:
                left = _resize_to_height(annotated_webcam, DISPLAY_HEIGHT)
                _click_state["left_w"] = left.shape[1]
                divider = np.full((DISPLAY_HEIGHT, 4, 3), 80, dtype=np.uint8)
                combined = np.concatenate([left, divider, right], axis=1)
                cv2.putText(combined, "Webcam", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                cv2.putText(combined, "Robot image", (left.shape[1] + 14, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                servo_status = f"fingers={'L+R' if len(fingertips)==2 else list(fingertips.keys()) or 'none'}  sam={'tracking' if sam_result else 'idle'}"
                cv2.putText(combined, servo_status, (left.shape[1] + 14, combined.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            else:
                _click_state["left_w"] = 0
                combined = right
                servo_status = f"fingers={'L+R' if len(fingertips)==2 else list(fingertips.keys()) or 'none'}  sam={'tracking' if sam_result else 'idle'}"
                cv2.putText(combined, servo_status, (10, combined.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()
        if pointer is not None:
            pointer.close()
        image_source.close()
        servo_publisher.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
