# webcam_mobilesam_point.py
import cv2
import numpy as np
from ultralytics import SAM

model = SAM("mobile_sam.pt")  # auto-downloads if needed

cap = cv2.VideoCapture(0)
clicked_point = None
last_mask = None

def on_mouse(event, x, y, flags, param):
    global clicked_point, last_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        last_mask = None

cv2.namedWindow("MobileSAM webcam")
cv2.setMouseCallback("MobileSAM webcam", on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    vis = frame.copy()

    if clicked_point is not None:
        x, y = clicked_point

        # Run SAM only when a new point is clicked, then reuse mask.
        if last_mask is None:
            results = model.predict(
                source=frame,
                points=[[x, y]],
                labels=[1],
                verbose=False
            )

            if results and results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                last_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

        if last_mask is not None:
            overlay = np.zeros_like(frame)
            overlay[last_mask > 0] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.45, 0)

        cv2.circle(vis, clicked_point, 5, (0, 0, 255), -1)

    cv2.putText(
        vis,
        "Left-click object | r reset | q quit",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("MobileSAM webcam", vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        clicked_point = None
        last_mask = None

cap.release()
cv2.destroyAllWindows()
