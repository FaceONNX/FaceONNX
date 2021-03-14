import cv2
from faceonnx import FaceDetectorLight
import numpy
import os

root = "images"
path = "results"

print("FaceONNX: Face detection")
files = os.listdir(root)

if (os.path.exists(path) is False):
    os.mkdir(path)

# Initialize FaceDetectorLight
faceDetectorLight = FaceDetectorLight(0.95, 0.25)
print(f"Processing {len(files)} images")

# Apply FaceDetectorLight to the files
# and draw inference results
for file in files:
    image = cv2.imread(os.path.join(root, file))
    boxes = faceDetectorLight.Forward(image)
    print(f"Image: [{file}] --> detected [{len(boxes)}] faces");

    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

    cv2.imwrite(os.path.join(path, file), image)

print("Done.")