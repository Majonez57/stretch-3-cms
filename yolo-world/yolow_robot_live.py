"""
Runs yolo-W with a given object to find live using stretch3's wrist camera

Press Q to quit.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from ultralytics import YOLOWorld

from vision.image_source import ZMQImageSource
from vision.fingertip_detector import FingertipDetector


WEBCAM_INDEX = 0
DISPLAY_HEIGHT = 480
CONFIRM_FRAMES = 60
STABILITY_RADIUS = 25
WINDOW_NAME = "Vision demo - press Q to quit"
USE_FINGER_POINTER = True

# Shared state for mouse click handler
_click_state: dict = {"pending": None, "left_w": 0, "robot_scale": 1.0}


def main() -> None:
    image_source = ZMQImageSource(host="192.168.239.2", port=4405)

    model = YOLOWorld("yolov8l-world.pt")

    objects = input("Enter Comma separated objects:").split(",")

    model.set_classes(objects)

    try:
        command_fired = False
        while True:
            
            robot_frame = image_source.get_frame()
            depth_frame = image_source.get_depth()
            depth_camera_info = image_source.get_depth_camera_info()
            camera_info = image_source.get_camera_info()
            depth_scale = image_source.get_depth_scale()

            # Use depth camera info for accuracy; fall back to colour
            active_camera_info = depth_camera_info or camera_info

            rh, rw = robot_frame.shape[:2]
            robot_display = robot_frame.copy()

            # --- Runs YOLO-W on the image with the given object
        
            cropped_frame = robot_frame.copy() #robot_frame[:, 80:560]
            print(cropped_frame.shape)

            yolo_results = model.predict(
                cropped_frame,
                imgsz=640,
                conf=0.2,
                #device='cpu'
            )

            found = False
            for result in yolo_results:
                if found:
                    break
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    cx = (x1+x2) /2
                    cy = (y1+y2) /2

                    cv2.circle(
                    robot_frame,
                    (int(cx), int(cy)),
                    10,
                    (0, 255, 0),
                    -1
                    )
                    cv2.rectangle(robot_frame,
                                    (int(x1),int(y1)),
                                    (int(x2),int(y2)),
                                    (0,255,0),2)
                    
            cv2.imshow('Yolo-W Detection. Press q to continue', robot_frame)
            cv2.waitKey(1)
            
    finally:
        
        image_source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
