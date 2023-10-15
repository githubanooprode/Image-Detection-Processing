from ultralytics import YOLO
import cv2
import cvzone
import math

#Footage Read from Camera in variable CAP
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 1920)

#Reading Footage and Detecting Under YOLO
model = YOLO("yolov8n.pt")


#Define Clasess for YOLO Model
classNames = ["Human", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "Mouse"
              ]

#Reading Loop for detecting a footage and recognising classes from footage.
while True:
    #Reading images from captured footage and detecting objects from it.
    success, img = cap.read()
    #Generating Detected images via YOLO and stored into results variable.
    results = model(img,stream=True)


    # r is result in results.
    for r in results:
        #This syntax used to draw a box for every result detected that is r.
        boxes = r.boxes
        #For every box which has been detected for evry r.
        for box in boxes:

            #Confidence
            #conf = math.ceil((box.conf[0]*100))/100
            conf = math.ceil(box.conf[0]*100)


            #ClassNames
            cls = int(box.cls[0])


            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if conf<=50 :
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            elif conf>=51 :
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)


            cvzone.putTextRect(img, f'{classNames[cls]} {conf} {"Accuracy"}', (max(0,x1), max(35,y1)))



    cv2.imshow("image", img)
    cv2.waitKey(1)
