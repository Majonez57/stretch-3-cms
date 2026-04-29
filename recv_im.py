import zmq
import cv2
import argparse


def main(use_remote_computer):
    print("cv2.__version__ =", cv2.__version__)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)

    if use_remote_computer:
        address = f"tcp://192.168.239.2:4405"
    else:
        address = f"tcp://127.0.0.1:4405"

    socket.connect(address)
    print(f"Receiving images from {address}")

    try:
        while True:
            d405_output = socket.recv_pyobj()
            color_image = d405_output["color_image"]

            cv2.imshow("Received Image", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Display Received D405 Images",
        description="Receives D405 color images over ZMQ and displays them in an OpenCV window.",
    )

    parser.add_argument(
        "-r",
        "--remote",
        action="store_true",
        help="Use this when running on a remote computer.",
    )

    args = parser.parse_args()
    main(args.remote)
