import math

import cvzone
from ultralytics import YOLO
import cv2
from sort import *

# tracking the objects using SORT
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

model = YOLO('/Users/shashidhar/PycharmProjects/Car_Count_YOLO_SORT/Yolo Weights/yolov8n.pt')

video = cv2.VideoCapture('/Users/shashidhar/PycharmProjects/Car_Count_YOLO_SORT/videos/video3.mp4')

total_cars = 0
while True:
    ret, frame = video.read()

    width = int(video.get(3)) # 3 will return the width property of the frame of the image
    height = int(video.get(4)) # 4 will return the height property of the frame of the image

    results = model(frame, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # confidence and class Name
            classIndex = int(box.cls[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # we are getting classIndex = 2 for car object
            if classIndex == 2 and confidence > 0.875:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cvzone.putTextRect(frame, f'{currentClass} {confidence}', (max(0, x1), max(35, y1)))

                currentDetection = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentDetection))

    trackerResults = tracker.update(detections)

    for result in trackerResults:
        x1, y1, x2, y2, id = result
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, int(x1)), max(35, int(y1))))
        total_cars = max(id, total_cars) # find the maximum all the ids of the cars and store in total_car

    cvzone.putTextRect(frame, f'Car count: {total_cars}', (50, 50))
    cvzone.putTextRect(frame, f'Press \'q\' to exit', (width - 450, height - 50))
    cv2.imshow('Image', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
