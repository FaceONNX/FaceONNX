import cv2
from faceonnx import FaceDetectorLight, FaceParser, Crop
import os

print("FaceONNX: Face semantic segmentation")
root = "images"
files = os.listdir(root)

# Initialize FaceDetectorLight and FaceParser
faceDetectorLight = FaceDetectorLight(0.95, 0.5)
faceParser = FaceParser()
print(f"Processing {len(files)} images")

# Apply FaceDetectorLight and FaceParser to the files
for file in files:
    image = cv2.imread(os.path.join(root, file))
    boxes = faceDetectorLight.Forward(image)
    print(f"Image: [{file}] --> detected [{len(boxes)}] faces");

    # Show segmentation maps
    for box in boxes:
        cropped = Crop(image, box)
        segmap = faceParser.Forward(cropped)
        cropped = faceParser.ToBitmap(segmap)

        while cv2.waitKey(1) < 0:
            cv2.imshow(f"Segmented face", cropped)

print("Done.")