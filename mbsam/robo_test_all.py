import cv2
import zmq
import yaml
import argparse
import numpy as np
from copy import deepcopy
from yaml.loader import SafeLoader

from ultralytics import SAM
from norfair import Detection, Tracker

import aruco_detector as ad
import aruco_to_fingertips as af
import d405_helpers_without_pyrealsense as dh


model = SAM("mobile_sam.pt")
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

clicked_point = None
current_mask = None


def draw_text(image, origin, text_lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    location = origin + np.array([0, -55])
    location = location.astype(np.int32)

    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_size, 4)
        (text_width, text_height), _ = text_size
        center = int(text_width / 2)
        offset = np.array([-center, i * (1.7 * text_height)]).astype(np.int32)

        cv2.putText(image, line, location + offset, font, font_size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, location + offset, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)


def segment_from_point(frame, point):
    x, y = point

    results = model.predict(
        source=frame,
        points=[[x, y]],
        labels=[1],
        verbose=False
    )

    if not results or results[0].masks is None:
        return None

    mask = results[0].masks.data[0].cpu().numpy()

    return cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


def mask_centroid(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return np.array([xs.mean(), ys.mean()])


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return {
        "min_x": float(xs.min()),
        "min_y": float(ys.min()),
        "max_x": float(xs.max()),
        "max_y": float(ys.max())
    }


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2)


def on_mouse(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


class SamServoPerception:
    def __init__(self):
        self.camera_info = None
        self.depth_scale = None

        self.marker_info = {}
        with open("aruco_marker_info.yaml") as f:
            self.marker_info = yaml.load(f, Loader=SafeLoader)

        self.aruco_detector = ad.ArucoDetector(
            marker_info=self.marker_info,
            show_debug_images=False,
            use_apriltag_refinement=False,
            brighten_images=False
        )

        self.fingertip_part = "cup_top"
        self.aruco_to_fingertips = af.ArucoToFingertips(
            default_height_above_mounting_surface=af.suctioncup_height[self.fingertip_part]
        )

    def set_camera_parameters(self, camera_info, depth_scale):
        self.camera_info = camera_info
        self.depth_scale = depth_scale

    def apply(self, color_image, depth_image):
        global clicked_point, current_mask

        assert (
            self.camera_info is not None and self.depth_scale is not None
        ), "ERROR: set_camera_parameters must be called before apply."

        self.aruco_detector.update(color_image, self.camera_info)
        markers = self.aruco_detector.get_detected_marker_dict()
        fingertips = self.aruco_to_fingertips.get_fingertips(markers)

        detections = []

        if clicked_point is not None:
            current_mask = segment_from_point(color_image, clicked_point)
            clicked_point = None

        if current_mask is not None:
            centroid = mask_centroid(current_mask)

            if centroid is not None:
                detections.append(Detection(points=centroid))

        tracked_objects = tracker.update(detections=detections)

        for obj in tracked_objects:
            cx, cy = obj.estimate[0].astype(int)
            current_mask = segment_from_point(color_image, (cx, cy))

        sam_output = []
        best_object = None

        if current_mask is not None:
            box = mask_to_box(current_mask)
            mask_polygon = mask_to_polygon(current_mask)

            if box is not None and mask_polygon is not None:
                box_min_x = box["min_x"]
                box_min_y = box["min_y"]
                box_max_x = box["max_x"]
                box_max_y = box["max_y"]

                box_width_x = box_max_x - box_min_x
                box_width_y = box_max_y - box_min_y
                max_box_side_pix = max(box_width_x, box_width_y)

                mask_min_x = int(box_min_x)
                mask_min_y = int(box_min_y)
                mask_max_x = int(box_max_x)
                mask_max_y = int(box_max_y)

                ball_depth_crop = depth_image[mask_min_y:mask_max_y, mask_min_x:mask_max_x]

                mask_crop = current_mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x]

                valid_depth_values = ball_depth_crop[mask_crop > 0]

                if valid_depth_values.size > 0:
                    estimated_depth = np.percentile(valid_depth_values, 50)
                    estimated_z_m = estimated_depth * self.depth_scale

                    center_pix = np.array([
                        (box_max_x + box_min_x) / 2.0,
                        (box_max_y + box_min_y) / 2.0
                    ])

                    left_side_pix = np.array([
                        box_min_x,
                        (box_max_y + box_min_y) / 2.0
                    ])

                    right_side_pix = np.array([
                        box_max_x,
                        (box_max_y + box_min_y) / 2.0
                    ])

                    width_pix = box_max_x - box_min_x

                    center_xyz = dh.pixel_to_3d(center_pix, estimated_z_m, self.camera_info)
                    center_ray = center_xyz / np.linalg.norm(center_xyz)

                    left_side_xyz = dh.pixel_to_3d(left_side_pix, estimated_z_m, self.camera_info)
                    right_side_xyz = dh.pixel_to_3d(right_side_pix, estimated_z_m, self.camera_info)

                    width_m = np.linalg.norm(right_side_xyz - left_side_xyz)

                    grasp_depth = width_m / 2.0
                    grasp_center_xyz = center_xyz + grasp_depth * center_ray

                    best_object = {
                        "name": "sam_tracked_object",
                        "max_box_side_pix": max_box_side_pix,
                        "confidence": 1.0,
                        "width_m": width_m,
                        "width_pix": width_pix,
                        "estimated_z_m": estimated_z_m,
                        "grasp_center_xyz": grasp_center_xyz,
                        "left_side_xyz": left_side_xyz,
                        "left_side_pix": left_side_pix,
                        "right_side_xyz": right_side_xyz,
                        "right_side_pix": right_side_pix,
                        "box": box,
                        "mask": mask_polygon
                    }

        if best_object is not None:
            minimal_object = deepcopy(best_object)

            del minimal_object["max_box_side_pix"]
            del minimal_object["mask"]
            del minimal_object["width_pix"]
            del minimal_object["left_side_pix"]
            del minimal_object["right_side_pix"]
            del minimal_object["box"]

            sam_output.append(minimal_object)

        send_dict = {
            "fingertips": fingertips,
            "yolo": sam_output
        }

        self.display_results(color_image, send_dict, best_object)

        return send_dict

    def display_results(self, color_image, send_dict, best_object):
        task_relevant_image = np.copy(color_image)

        self.aruco_to_fingertips.draw_fingertip_frames(
            send_dict["fingertips"],
            task_relevant_image,
            self.camera_info,
            axis_length_in_m=0.02,
            draw_origins=True,
            write_coordinates=True
        )

        if best_object is not None:
            mask = best_object["mask"]

            brighten_mask = np.ones(task_relevant_image.shape[:2], np.float32)
            cv2.polylines(brighten_mask, [mask.astype(np.int32)], True, 4.0, 3, lineType=cv2.LINE_AA)
            cv2.fillPoly(brighten_mask, [mask.astype(np.int32)], 2.0, lineType=cv2.LINE_AA)

            for channel in range(3):
                task_relevant_image[:, :, channel] = np.minimum(
                    brighten_mask * task_relevant_image[:, :, channel],
                    255
                ).astype(np.uint8)

            grasp_center_xy = dh.pixel_from_3d(
                best_object["grasp_center_xyz"],
                self.camera_info
            )

            grasp_point = grasp_center_xy.astype(np.int32)

            cv2.circle(
                task_relevant_image,
                grasp_point,
                6,
                (255, 0, 0),
                -1,
                lineType=cv2.LINE_AA
            )

            x, y, z = best_object["grasp_center_xyz"] * 100.0

            text_lines = [
                "{:.1f} cm wide".format(best_object["width_m"] * 100.0),
                "{:.1f}, {:.1f}, {:.1f} cm".format(x, y, z)
            ]

            draw_text(task_relevant_image, grasp_point, text_lines)

        cv2.imshow("Task Relevant Results", task_relevant_image)


def main(use_remote_computer):
    print("cv2.__version__ =", cv2.__version__)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)

    if use_remote_computer:
        address = "tcp://192.168.239.2:4405"
    else:
        address = "tcp://127.0.0.1:4405"

    socket.connect(address)
    print(f"Receiving images from {address}")

    perception = SamServoPerception()

    window_name = "SAM Servo Perception"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            d405_output = socket.recv_pyobj()

            color_image = d405_output["color_image"]
            depth_image = d405_output["depth_image"]

            if perception.camera_info is None:
                perception.set_camera_parameters(
                    d405_output["depth_camera_info"],
                    d405_output["depth_scale"]
                )

            send_dict = perception.apply(color_image, depth_image)

            cv2.imshow(window_name, color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SAM Servo Perception over TCP",
        description="Receives D405 images over ZMQ and returns fingertip + SAM-tracked object data.",
    )

    parser.add_argument(
        "-r",
        "--remote",
        action="store_true",
        help="Use this when running on a remote computer.",
    )

    args = parser.parse_args()
    main(args.remote)
