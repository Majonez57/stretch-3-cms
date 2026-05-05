import cv2
import numpy as np
from ultralytics import SAM
from norfair import Detection, Tracker

model = SAM("mobile_sam.pt")
cap = cv2.VideoCapture(0)

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
    return cv2.resize(mask.astype(np.uint8),
                      (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])

def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("Norfair + MobileSAM")
cv2.setMouseCallback("Norfair + MobileSAM", on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    vis = frame.copy()
    detections = []

    # Run SAM only when user clicks OR occasionally
    if clicked_point is not None:
        current_mask = segment_from_point(frame, clicked_point)
        clicked_point = None

    if current_mask is not None:
        c = mask_centroid(current_mask)
        if c is not None:
            detections.append(Detection(points=c))

    tracked_objects = tracker.update(detections=detections)

    # Use tracked position to refresh SAM occasionally
    for obj in tracked_objects:
        cx, cy = obj.estimate[0].astype(int)

        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

        # OPTIONAL: refresh SAM at tracked position
        current_mask = segment_from_point(frame, (cx, cy))

    if current_mask is not None:
        overlay = np.zeros_like(frame)
        overlay[current_mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)

    cv2.imshow("Norfair + MobileSAM", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
