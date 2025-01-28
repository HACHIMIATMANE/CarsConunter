from sympy import im
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np

from sort import Sort  #The tracker is imported from the sort module

#cap = cv2.VideoCapture(1)  # For Webcam
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture(r"Project1_Car_Counter\cars.mp4")  # For Video


model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#mask = cv2.imread('Project1_Car_Counter\mask.png')
#mask = cv2.resize(mask, (1280, 720))a

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img,mask)
    results = model(img, stream=True)
    
    imgGraphics = cv2.imread("Project1_Car_Counter\image.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            #cvzone.cornerRect(img, (x1, y1, w, h),l=5) # l = 5 is the thickness of the corner rectangle
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=2, thickness=3, offset=5)
            #offset is the distance of the text from the rectangle

            currentClass = classNames[cls]
            if currentClass == "car" or "truck" or "motorbike" or "bus" and conf > 0.3:
                #cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=5)
                #cvzone.cornerRect(img, (x1, y1, w, h),l=5)
                currentArray = np.array([x1, y1, x2, y2, conf]) #creating an array of the coordinates and confidence
                detections = np.vstack((detections, currentArray)) #stacking the currentArray to detections
 
    

    resultsTrucker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for track in resultsTrucker:
        x1, y1, x2, y2, id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
        print(track)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h),l=5, rt=5, colorR=(255, 0,0))
        #cvzone.putTextRect(img, f'Count: {int(totalCount)}', (50,50))
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #cvzone.putTextRect(img, f'ID: {id}', (x1, y1), scale=2, thickness=3, offset=5)

        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f'Count: {int(len(totalCount))}', (50,50))

    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1) #this command how long the image will be displayed : 1 means 1ms and 0 means the image will be displayed until the user closes it