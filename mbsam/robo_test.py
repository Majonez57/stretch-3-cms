import cv2
import zmq
import argparse
import numpy as np
from ultralytics import SAM
from norfair import Detection, Tracker


model = SAM("mobile_sam.pt")
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

clicked_point = None
current_mask = None


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


def on_mouse(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main(use_remote_computer):
    global clicked_point, current_mask

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

    window_name = "Norfair + MobileSAM"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            d405_output = socket.recv_pyobj()
            frame = d405_output["color_image"]

            if frame is None:
                continue

            vis = frame.copy()
            detections = []

            if clicked_point is not None:
                current_mask = segment_from_point(frame, clicked_point)
                clicked_point = None

            if current_mask is not None:
                c = mask_centroid(current_mask)

                if c is not None:
                    detections.append(Detection(points=c))

            tracked_objects = tracker.update(detections=detections)

            for obj in tracked_objects:
                cx, cy = obj.estimate[0].astype(int)

                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

                current_mask = segment_from_point(frame, (cx, cy))

            if current_mask is not None:
                overlay = np.zeros_like(frame)
                overlay[current_mask > 0] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)

            cv2.imshow(window_name, vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Norfair + MobileSAM over TCP",
        description="Receives D405 color images over ZMQ and runs MobileSAM + Norfair tracking.",
    )

    parser.add_argument(
        "-r",
        "--remote",
        action="store_true",
        help="Use this when running on a remote computer.",
    )

    args = parser.parse_args()
    main(args.remote)
