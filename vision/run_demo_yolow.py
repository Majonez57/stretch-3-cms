"""
Runs yolo-W with a given object to find live using stretch3's wrist camera
Uses OpenCV's Legacy Tracker (MOSSE) to track the object when YOLO drops out.

Press Q to quit.
"""

from __future__ import annotations

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from ultralytics import YOLOWorld, SAM

from vision.image_source import ZMQImageSource
from vision.servo_publisher import ServoPublisher
from vision.fingertip_detector import FingertipDetector

WEBCAM_INDEX = 0
DISPLAY_HEIGHT = 480
CONFIRM_FRAMES = 60
STABILITY_RADIUS = 25
WINDOW_NAME = "Vision demo - press Q to quit"
USE_FINGER_POINTER = True

# Configuration: How often to run YOLO.
# 0 = Every frame
# 5 = Every 5th frame (Tracker fills in the gaps)
YOLO_RATE = 0 

_click_state: dict = {"pending": None, "left_w": 0, "robot_scale": 1.0}

def _pixel_to_3d(uv: list[float], depth_m: float, camera_info: dict) -> np.ndarray:
    K = camera_info["camera_matrix"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([
        (uv[0] - cx) * depth_m / fx,
        (uv[1] - cy) * depth_m / fy,
        depth_m,
    ])


def main() -> None:
    image_source = ZMQImageSource(host="192.168.239.2", port=4405)
    servo_publisher = ServoPublisher(port=4010)
    fingertip_detector = FingertipDetector()
    model = YOLOWorld("yolov8l-world.pt")
    sam = SAM("mobile_sam.pt")
    objects = input("Enter Comma separated objects:").split(",")
    model.set_classes(objects)

    # Initialize OpenCV Tracker
    tracker = cv2.legacy.TrackerCSRT_create()
    
    tracker_initialized = False
    frame_count = 0

    try:
        command_fired = False
        while True:
            frame_count += 1
            robot_frame = image_source.get_frame()
            depth_frame = image_source.get_depth()
            depth_camera_info = image_source.get_depth_camera_info()
            camera_info = image_source.get_camera_info()
            depth_scale = image_source.get_depth_scale()

            active_camera_info = depth_camera_info or camera_info
            rh, rw = robot_frame.shape[:2]
            
            # Use a clean copy for processing/detection
            cropped_frame = robot_frame.copy() 

            fingertips: dict = {}
            if active_camera_info is not None:
                try:
                    fingertips = fingertip_detector.detect(robot_frame, active_camera_info)
                except Exception:
                    pass
            
            # --- Logic Control ---
            run_yolo = (YOLO_RATE == 0) or (frame_count % YOLO_RATE == 0)
            
            final_bbox = None # [x1, y1, x2, y2]
            found = False
            source_type = "None"

            # 1. Try YOLO
            if run_yolo:
                t0 = time.time()
                yolow_results = model.predict(
                    cropped_frame,
                    conf=0.2,
                    verbose=False
                )
                boxes = (yolow_results[0]).boxes
                
                if boxes is not None and len(boxes) > 0:
                    boxes_xyxy = boxes.xyxy
                    confs = boxes.conf
                    top_conf_idx = confs.argmax() 
                    
                    best_bbox = boxes_xyxy[top_conf_idx].tolist()
                    
                    # YOLO found it. We will use this bbox.
                    final_bbox = best_bbox
                    found = True
                    source_type = "YOLO"

                    # Re-initialize the tracker with this fresh box
                    # This corrects any drift from previous tracking frames
                    x1, y1, x2, y2 = map(int, best_bbox)
                    # Tracker expects (x, y, w, h)
                    tracker.init(cropped_frame, (x1, y1, x2-x1, y2-y1))
                    tracker_initialized = True

            # 2. Fallback to Tracker
            if not found and tracker_initialized:
                success, box = tracker.update(cropped_frame)
                if success:
                    # Tracker returned (x, y, w, h). Convert to (x1, y1, x2, y2)
                    x, y, w, h = map(int, box)
                    final_bbox = [x, y, x+w, y+h]
                    found = True
                    source_type = "Tracker"
                else: print("[WARN]: Tracker Failed")

            # --- Processing ---
            if found and final_bbox is not None:
                x1, y1, x2, y2 = map(int, final_bbox)
                
                # Boundary check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(rw, x2), min(rh, y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    cv2.imshow('Vision Demo', robot_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Visualization
                color = (0, 255, 0) if source_type == "YOLO" else (0, 165, 255)
                cv2.rectangle(robot_frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(robot_frame, source_type, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Run SAM
                try:
                    t_sam = time.time()
                    # SAM expects [[x1, y1, x2, y2]]
                    sam_results = sam(cropped_frame, bboxes=[[x1-5, y1-5, x2+5, y2+5]], verbose=False)
                    t1 = time.time()
                    print(f"Source: {source_type}, SAM time: {(t1-t_sam)*1000:.1f}ms")

                    if sam_results[0].masks is not None:
                        mask = sam_results[0].masks.data[0].cpu().numpy()
                        robot_frame[mask] = (255, 0, 0)
                        if source_type == "Tracker":
                            # If tracker detection, we update the bounding box to instead match the mask
                            rows = np.any(mask, axis=1)
                            cols = np.any(mask, axis=0)
                            
                            rmin, rmax = np.where(rows)[0][[0, -1]]
                            cmin, cmax = np.where(cols)[0][[0, -1]]
                            x1,y1,x2,y2 = int(cmin), int(rmin), int(cmax), int(rmax)

                            color = (30, 134, 10)
                            cv2.rectangle(robot_frame, (x1, y1), (x2, y2), color, 1)
                            cv2.putText(robot_frame, "SAM", (x2-10, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                            tracker.init(cropped_frame, (x1,y1,x2-x1,y2-y1))
                    else:
                        # SAM failed to segment the tracked box
                        found = False
                except Exception as e:
                    print(f"SAM error: {e}")
                    found = False

            cv2.imshow('Vision Demo', robot_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if not found: continue

            # --- Depth and Servoing ---
            if y2 <= y1 or x2 <= x1: continue

            try:
                depth_crop = depth_frame[y1:y2, x1:x2]
                mask_crop = mask[y1:y2, x1:x2]
                
                if mask_crop.shape != depth_crop.shape: continue

                valid_depths = depth_crop[mask_crop > 0]
                valid_depths = valid_depths[valid_depths > 0]
                depth_scale = depth_scale if depth_scale is not None else 0.001

                yolow_result = {
                    "mask"       : mask,
                    "center_pix" : (cx, cy)
                }

                if valid_depths.size > 0:
                    estimated_z_m = float(np.percentile(valid_depths, 50)) * depth_scale
                    center_xyz = _pixel_to_3d([cx, cy], estimated_z_m, camera_info)
                    left_xyz = _pixel_to_3d([x1, cy], estimated_z_m, camera_info)
                    right_xyz = _pixel_to_3d([x2, cy], estimated_z_m, camera_info)
                    width_m = float(np.linalg.norm(right_xyz - left_xyz))

                    center_ray = center_xyz / np.linalg.norm(center_xyz)
                    grasp_center_xyz = center_xyz + (width_m / 2.0) * center_ray

                    yolow_result.update({
                        "width_m"          : width_m,
                        "estimated_z_m"    : estimated_z_m,
                        "grasp_center_xyz" : grasp_center_xyz,
                        "left_side_xyz"    : left_xyz,
                        "right_side_xyz"   : right_xyz
                    })

                servo_publisher.publish(fingertips, yolow_result)
            
            except Exception as e:
                print(f"Processing error: {e}")
                continue
            
    finally:
        image_source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()