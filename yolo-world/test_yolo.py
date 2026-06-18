from ultralytics import YOLOWorld
import cv2

model = YOLOWorld("yolov8s-world.pt")

# Things you want to detect
model.set_classes(["lemon"])

results = model.predict(
    "food.png",
    imgsz=560,
    conf=0.2,
    #device="cpu"
)

img = cv2.imread("food.png")


for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cv2.circle(
            img,
            (int(cx), int(cy)),
            10,
            (0, 255, 0),
            -1
        )

        print(f"Center: ({cx:.1f}, {cy:.1f})")
    
cv2.imwrite("result.png", img)