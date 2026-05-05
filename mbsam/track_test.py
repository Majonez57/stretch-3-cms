import cv2
import numpy as np
from ultralytics import SAM

model = SAM("mobile_sam.pt")
cap = cv2.VideoCapture(0)

clicked_point = None
tracker = None
tracking = False

current_mask = None
current_bbox = None
frame_idx = 0
SAM_EVERY_N_FRAMES = 10

def make_tracker():
    for name in ["TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"]:
        if hasattr(cv2, name):
            return getattr(cv2, name)()
    if hasattr(cv2, "legacy"):
        for name in ["TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"]:
            if hasattr(cv2.legacy, name):
                return getattr(cv2.legacy, name)()
    raise RuntimeError("No tracker found. Try: pip install opencv-contrib-python")

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

def bbox_center(bbox):
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)

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
    mask = cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return mask

def init_tracker_from_mask(frame, mask):
    global tracker, tracking, current_bbox

    bbox = mask_to_bbox(mask)
    if bbox is None:
        tracker = None
        tracking = False
        current_bbox = None
        return

    tracker = make_tracker()
    tracker.init(frame, bbox)
    tracking = True
    current_bbox = bbox

def on_mouse(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("MobileSAM refreshed segment tracking")
cv2.setMouseCallback("MobileSAM refreshed segment tracking", on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1
    vis = frame.copy()

    # New object selected
    if clicked_point is not None:
        mask = segment_from_point(frame, clicked_point)
        clicked_point = None

        if mask is not None:
            current_mask = mask
            init_tracker_from_mask(frame, current_mask)

    # Track between SAM refreshes
    tracked_center = None

    if tracking and tracker is not None:
        ok, bbox = tracker.update(frame)

        if ok:
            current_bbox = tuple(map(int, bbox))
            tracked_center = bbox_center(current_bbox)
        else:
            tracking = False
            tracker = None
            current_bbox = None

    # Re-run SAM at tracked center for a fresh accurate mask
    if (
        tracking
        and tracked_center is not None
        and frame_idx % SAM_EVERY_N_FRAMES == 0
    ):
        refreshed_mask = segment_from_point(frame, tracked_center)

        if refreshed_mask is not None:
            current_mask = refreshed_mask
            init_tracker_from_mask(frame, current_mask)

    # Draw current accurate mask
    if current_mask is not None:
        overlay = np.zeros_like(frame)
        overlay[current_mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.45, 0)

        contours, _ = cv2.findContours(
            current_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    if current_bbox is not None:
        x, y, w, h = current_bbox
        cx, cy = bbox_center(current_bbox)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

    cv2.putText(
        vis,
        f"Click object | SAM every {SAM_EVERY_N_FRAMES} frames | r reset | q quit",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("MobileSAM refreshed segment tracking", vis)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        clicked_point = None
        tracker = None
        tracking = False
        current_mask = None
        current_bbox = None

cap.release()
cv2.destroyAllWindows()
