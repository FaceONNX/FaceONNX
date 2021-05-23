import cv2
from faceonnx.imaging import Crop
import numpy as np
from faceonnx import FaceDetectorLight, FaceLandmarksExtractor, FaceMaskClassifier, landmarks

# Initialize FaceDetectorLight and FaceLandmarksExtractor
faceDetectorLight = FaceDetectorLight(0.95, 0.25)
faceLandmarksExtractor = FaceLandmarksExtractor()
faceMaskClassifier = FaceMaskClassifier()

# Initialize webcam feed
color = (255, 255, 0)
video = cv2.VideoCapture(0)

# Start webcam
while True:

    # Get frame and apply 
    # FaceDetectorLight, FaceLandmarksExtractor and FaceMaskClassifier
    ret, frame = video.read()
    boxes = faceDetectorLight.Forward(frame)

    for i in range(len(boxes)):
        box = boxes[i]
        cropped = Crop(frame, box)
        landmarks = faceLandmarksExtractor.Forward(cropped)
        aligned = FaceLandmarksExtractor.Align(cropped, landmarks)
        mask = faceMaskClassifier.Forward(aligned)
        label = FaceMaskClassifier.Labels[np.argmax(mask)]

        # Draw inference results
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 4)
        cv2.putText(frame, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    cv2.imshow('FaceONNX: Face mask detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
