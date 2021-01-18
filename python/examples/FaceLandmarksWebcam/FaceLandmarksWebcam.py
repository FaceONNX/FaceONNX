import cv2
from faceonnx import FaceDetectorLight, FaceLandmarksExtractor
import numpy
import os
import onnxruntime as ort

faceDetectorLight = FaceDetectorLight(0.95, 0.25)
faceLandmarksExtractor = FaceLandmarksExtractor()

# Initialize webcam feed
color = (255, 255, 0)
video = cv2.VideoCapture(0)

while(True):

    ret, frame = video.read()
    boxes = faceDetectorLight.Forward(frame)
    landmarks = faceLandmarksExtractor.Forward(frame, boxes)

    for i in range(len(boxes)):
        box = boxes[i]
        landmark = landmarks[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 4)
        
        for (x, y) in landmark:
            cv2.circle(frame, (x, y), 5, color)

    cv2.imshow('FaceONNX: Face landmarks extraction (webcamera test)', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
