# Object_tracking_and_instance_segmentation_of_people

AIM: Training a deep learning (from ultralytics import YOLOV8) model to perform an instance segmentation model with each instance labeled with one single color.

##Explanation of the code used in (main.py):
The code has two files namely main.py and tracker.py is used for processing a video of people walking, detecting people in each frame, tracking their movement, and saving the output as a new video. Here's a simple explanation of what the code does:

1.	Importing Libraries: The code starts by importing necessary libraries. These include OpenCV for video processing, pandas for data manipulation, and Ultralytics YOLO for object detection.

2.	Initializing YOLO Model: It initializes the YOLO object detection model (YOLOv8) from Ultralytics using a pre-trained model file called 'yolov8s.pt'.

3.	Setting up Event Handling: The code sets up an event handler for the 'RGB' window to capture mouse movements.

4.	Opening Video: It opens the input video file called 'background video _ people _ walking.mp4' using OpenCV's VideoCapture.

5.	Loading Class Labels: Class labels for object detection are loaded from a file called 'coco.txt'. The coca dataset has 80 different object classes these labels help identify the type of objects detected, like "person," "car," etc.

6.	Initializing Tracking: An object tracker (likely a custom module) is initialized to track the movement of people across frames.

7.	Initializing Video Output: It initializes a VideoWriter to save the processed frames as an output video in the 'mp4' format. The output video is named 'output_video.mp4', and it's set to have a frame rate of 20 frames per second and a frame size of 1020x500 pixels.

8.	Processing Video Frames: The code enters a loop to process each frame of the input video. It resizes each frame to a consistent size (1020x500 pixels) for processing.

9.	Object Detection: YOLO is used to detect objects in the frame. Detected objects are filtered to only include "person" objects, and their positions (bounding boxes) are collected.

10.	Object Tracking: The collected bounding boxes for people are passed to the object tracker, which helps maintain the identity and track their movement across frames.

11.	Drawing Bounding Boxes and Labels: The code draws bounding boxes around detected people and adds a label "person" near each box.

12.	Saving Processed Frames: The processed frame, with bounding boxes and labels, is written to the output video.

13.	Displaying Processed Frame: The processed frame is displayed in the 'RGB' window using OpenCV's imshow.

14.	Exiting the Loop: The loop continues until the end of the input video. You can exit the loop by pressing the 'Esc' key (ASCII code 27).

15.	Finalization: After processing all frames, the output video is released, the input video is closed, and all OpenCV windows are destroyed.

In summary, this code takes a video of people walking, detects and tracks the people, adds labels and bounding boxes to them, and saves the result as a output video .
