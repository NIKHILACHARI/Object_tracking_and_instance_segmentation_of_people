import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('background video _ people _ walking.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
persondown = {}
tracker = Tracker()
counter1 = []

personup = {}
counter2 = []
cy1 = 194
cy2 = 220
offset = 6

# Initialize the VideoWriter with the 'mp4v' codec for mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 format
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1020, 500))  # Adjust the output file name, frame rate, and frame size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255),2)

        # Add the label "person" to the bounding box
        cv2.putText(frame, "person", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)



    # Write the frame to the output video
    out.write(frame)

    cv2.imshow("RGB", frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the VideoWriter and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()

