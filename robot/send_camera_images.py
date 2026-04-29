"""
Robot-side D405 camera publisher.

Captures colour and depth frames from the D405 wrist camera and publishes
them over ZMQ so the dev machine can subscribe via ZMQImageSource.

Also detects ArUco markers (DICT_6X6_250) on each frame and includes their
pixel centroids in the published dict. This acts as a placeholder target for
testing the pointing pipeline before SAM is wired in. Remove the ArUco block
(marked below) once SAM is integrated.

Run on the Stretch robot:
    python robot/send_camera_images.py
    python robot/send_camera_images.py --local     # bind to localhost only
    python robot/send_camera_images.py --exposure medium
    python robot/send_camera_images.py --no-aruco  # disable ArUco detection

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import zmq

# Port must match CAMERA_ZMQ_PORT in config.py on the dev machine
ZMQ_PORT = 4405

# D405 capture resolution and framerate
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 15

EXPOSURE_KEYWORDS = ["low", "medium", "auto"]
_DEFAULT_EXPOSURE = 33000


def _start_pipeline(exposure: str) -> tuple[rs.pipeline, rs.pipeline_profile]:
    ctx = rs.context()
    devices = [
        {"name": d.get_info(rs.camera_info.name),
         "serial": d.get_info(rs.camera_info.serial_number)}
        for d in ctx.devices
    ]

    d405 = next((d for d in devices if d["name"].endswith("D405")), None)
    if d405 is None:
        print("ERROR: D405 camera not found. Is it connected?")
        sys.exit(1)

    print(f"D405 found: {d405['name']} ({d405['serial']})")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(d405["serial"])
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FRAME_FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
    profile = pipeline.start(config)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
    if exposure == "auto":
        sensor.set_option(rs.option.enable_auto_exposure, True)
    else:
        if exposure == "low":
            value = _DEFAULT_EXPOSURE // 3
        elif exposure == "medium":
            value = 30000
        else:
            value = int(exposure)
        sensor.set_option(rs.option.exposure, value)

    return pipeline, profile


def _get_camera_info(frame: rs.frame) -> dict:
    intrinsics = rs.video_stream_profile(frame.profile).get_intrinsics()
    camera_matrix = np.array([
        [intrinsics.fx, 0.0,           intrinsics.ppx],
        [0.0,           intrinsics.fy, intrinsics.ppy],
        [0.0,           0.0,           1.0],
    ])
    return {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": np.array(intrinsics.coeffs),
    }


# ---------------------------------------------------------------------------
# ArUco placeholder — remove this block once SAM is integrated
# ---------------------------------------------------------------------------
def _make_aruco_detector() -> aruco.ArucoDetector:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco_dict, params)


def _detect_aruco(detector: aruco.ArucoDetector, color_image: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return {marker_id: (cx, cy)} for all visible markers."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    result = {}
    for corner, marker_id in zip(corners, ids.flatten()):
        cx = int(corner[0, :, 0].mean())
        cy = int(corner[0, :, 1].mean())
        result[int(marker_id)] = (cx, cy)
    return result
# ---------------------------------------------------------------------------


def main(local_only: bool, exposure: str, use_aruco: bool) -> None:
    pipeline, profile = _start_pipeline(exposure)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale}")

    # ArUco placeholder — remove once SAM is integrated
    aruco_detector = _make_aruco_detector() if use_aruco else None

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)

    address = f"tcp://127.0.0.1:{ZMQ_PORT}" if local_only else f"tcp://*:{ZMQ_PORT}"
    socket.bind(address)
    print(f"Publishing on {address}  (Ctrl+C to stop)")

    camera_info_cached = None
    frames_sent = 0
    start = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            if camera_info_cached is None:
                camera_info_cached = {
                    "color": _get_camera_info(color_frame),
                    "depth": _get_camera_info(depth_frame),
                }

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ArUco placeholder — remove once SAM is integrated
            aruco_markers = (
                _detect_aruco(aruco_detector, color_image)
                if aruco_detector is not None
                else {}
            )

            socket.send_pyobj({
                "color_image":       color_image,
                "depth_image":       depth_image,
                "color_camera_info": camera_info_cached["color"],
                "depth_camera_info": camera_info_cached["depth"],
                "depth_scale":       depth_scale,
                "aruco_markers":     aruco_markers,  # {id: (cx, cy)} — remove with SAM
            })

            frames_sent += 1
            if frames_sent % 150 == 0:
                elapsed = time.time() - start
                print(f"Sent {frames_sent} frames  ({frames_sent / elapsed:.1f} fps)")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pipeline.stop()
        socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream D405 frames over ZMQ.")
    parser.add_argument(
        "--local", action="store_true",
        help="Bind to localhost only (no remote access).",
    )
    parser.add_argument(
        "--exposure", default="low",
        choices=EXPOSURE_KEYWORDS + ["<integer>"],
        help="Exposure setting: low, medium, auto, or an integer value.",
    )
    parser.add_argument(
        "--no-aruco", action="store_true",
        help="Disable ArUco marker detection.",
    )
    args = parser.parse_args()
    main(local_only=args.local, exposure=args.exposure, use_aruco=not args.no_aruco)
