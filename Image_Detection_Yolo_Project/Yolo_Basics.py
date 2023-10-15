from ultralytics import YOLO
import cv2

model = YOLO(yolov8l.pt)
result = model("Sources/images/pic_1.jpg", show=True)
cv2.waitKey(1)
