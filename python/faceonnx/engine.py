import onnxruntime
import numpy as np
import cv2
import os
from .imaging import Rotate, Resize, Crop, ToBox
from .landmarks import GetMeanPoint, GetSupportPoint, GetAngle, GetLeftEye, GetRightEye

# main
this = os.path.dirname(__file__)
models_path = os.path.join(this, "models")

class FaceDetectorLight:

    def __init__(self, confidenceThreshold = 0.95, nmsThreshold = 0.5, sessionOptions = None):
        """
        Initializes face detector.
        Args:
            confidenceThreshold: Confidence threshold (defaults to 0.95)
            nmsThreshold: NonMaxSuppression threshold (defaults to 0.5)
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "face_detector_320.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name
        self.confidenceThreshold = confidenceThreshold
        self.nmsThreshold = nmsThreshold

    def Forward(self, image):
        """
        Returns face detection results.
        Args:
            image: Bitmap

        Returns:
            Rectangles
        """
        input_name = self.session.get_inputs()[0].name
        h, w, _ = image.shape
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 240))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        confidences, boxes = self.session.run(None, {self.input_name: img})

        from .internal import Detector
        rectangles, labels, probs = Detector(w, h, confidences, boxes, self.confidenceThreshold, self.nmsThreshold)

        # scale to face box
        for i in range(len(rectangles)):
            rectangles[i] = ToBox(rectangles[i])

        return rectangles

class FaceDetector:

    def __init__(self, confidenceThreshold = 0.95, nmsThreshold = 0.5, sessionOptions = None):
        """
        Initializes face detector.
        Args:
            confidenceThreshold: Confidence threshold (defaults to 0.95)
            nmsThreshold: NonMaxSuppression threshold (defaults to 0.5)
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "face_detector_640.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name
        self.confidenceThreshold = confidenceThreshold
        self.nmsThreshold = nmsThreshold

    def Forward(self, image):
        """
        Returns face detection results.
        Args:
            image: Bitmap

        Returns:
            Rectangles
        """
        h, w, _ = image.shape
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        confidences, boxes = self.session.run(None, {self.input_name: img})

        from .internal import Detector
        boxes, labels, probs = Detector(w, h, confidences, boxes, self.confidenceThreshold, self.nmsThreshold)

        # scale to face box
        for i in range(len(boxes)):
            boxes[i] = ToBox(boxes[i])

        return boxes

class FaceAgeClassifier:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = ["(0-2)", "(3-7)", "(8-14)", "(15-24)", "(25-37)", "(38-47)", "(48-59)", "(60-100)"]

    def __init__(self, sessionOptions = None):
        """
        Initializes face age classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "age_googlenet.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                outputs.append(self.__Forward(cropped))

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        img = Resize(image, (224, 224))
        img_mean = np.array([104, 117, 123])
        img = img - img_mean
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return self.session.run(None, {self.input_name: img})[0][0]

class FaceBeautyClassifier:

    def __init__(self, sessionOptions = None):
        """
        Initializes face beauty classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "beauty_resnet18.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                outputs.append(self.__Forward(cropped))

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        img = Resize(image, (224, 224))
        img_mean = np.array([104, 117, 123])
        img = (img - img_mean) / 255.0
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return round(2.0 * sum(self.session.run(None, {self.input_name: img})[0][0]), 1)

class FaceEmbedder:

    def __init__(self, sessionOptions = None):
        """
        Initializes face embedder.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "recognition_resnet27.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                outputs.append(self.__Forward(cropped))

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        img = Resize(image, (128, 128))
        img_mean = np.array([127.5, 127.5, 127.5])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return self.session.run(None, {self.input_name: img})[0][0]

class FaceEmotionClassifier:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disguest", "fear"]

    def __init__(self, sessionOptions = None):
        """
        Initializes face emotion classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "emotion_cnn.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                outputs.append(self.__Forward(cropped))

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        img = Resize(image, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 256.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return np.exp(self.session.run(None, {self.input_name: img})[0][0])

class FaceRaceClassifier:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = ["White", "Black", "Asian", "Indian"]

    def __init__(self, sessionOptions = None):
        """
        Initializes face race classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "race_googlenet.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                outputs.append(self.__Forward(cropped))

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        img = Resize(image, (224, 224))
        img_mean = np.array([104, 117, 123])
        img = img - img_mean
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return self.session.run(None, {self.input_name: img})[0][0]

class FaceGenderClassifier:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = ["Male", "Female"]

    def __init__(self, sessionOptions = None):
        """
        Initializes face gender classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "gender_googlenet.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                outputs.append(self.__Forward(cropped))

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        img = Resize(image, (224, 224))
        img_mean = np.array([104, 117, 123])
        img = img - img_mean
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return self.session.run(None, {self.input_name: img})[0][0]

class FaceLandmarksExtractor:

    def __init__(self, sessionOptions = None):
        """
        Initializes face landmarks extractor.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "landmarks_68_pfld.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                points = self.__Forward(cropped)

                for i in range(len(points)):
                    points[i] += (rectangle[0], rectangle[1])

                outputs.append(points)

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        h, w, _ = image.shape
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Resize(img, (112, 112))
        img = img / 255.0
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        output = self.session.run(None, {self.input_name: img})[0][0]
        points = output.reshape(-1, 2) * (w, h)
        return points.astype(np.int32)

    def Align(self, image, points):
        """[summary]

        Args:
            image ([type]): [description]
            points ([type]): [description]

        Returns:
            [type]: [description]
        """
        left = GetMeanPoint(GetLeftEye(points))
        right = GetMeanPoint(GetRightEye(points))
        point = GetSupportPoint(left, right)
        angle = GetAngle(left, right, point)
        return Rotate(image, angle)

class FaceParser:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = np.array([
                    (0,  0,  0),
                    (204, 0,  0),
                    (76, 153, 0),
                    (204, 204, 0),
                    (51, 51, 255),
                    (204, 0, 204),
                    (0, 255, 255),
                    (51, 255, 255),
                    (102, 51, 0),
                    (255, 0, 0),
                    (102, 204, 0),
                    (255, 255, 0),
                    (0, 0, 153),
                    (0, 0, 204),
                    (255, 51, 153), 
                    (0, 204, 204),
                    (0, 51, 0),
                    (255, 153, 51),
                    (0, 204, 0)], 
                    dtype=np.uint8)

    def __init__(self, sessionOptions = None):
        """
        Initializes face parser.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "face_unet_512.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                points = self.__Forward(cropped)

                for i in range(len(points)):
                    points[i] += (rectangle[0], rectangle[1])

                outputs.append(points)

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        h, w, _ = image.shape
        size = (512, 512)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Resize(img, size)
        img = img / 255.0
        img_mean = np.array([0.5, 0.5, 0.5])
        img = (img - img_mean) / img_mean
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # post-processing
        confidences = self.session.run(None, {self.input_name: img})[0][0]
        s = confidences.shape[0]
        maximum = np.max(confidences)
        minimum = np.min(confidences)
        probabilities = (confidences - minimum) / (maximum - minimum)

        return probabilities

    def ToBitmap(self, masks):
        """
        Returns bitmap from masks.
        Args:
            masks: Masks

        Returns:
            image: Bitmap
        """
        s, h, w = masks.shape
        image = np.zeros((h, w, 3))

        for y in range(h):
            for x in range(w):
                maximum = -2147483648
                index = 0

                for i in range(s):
                    if (masks[i, y, x] > maximum):
                        maximum = masks[i, y, x]
                        index = i
                
                color = self.Labels[index]
                image[y, x] = color[::-1]

        return image

class FaceParserLight:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = np.array([
                    (0,  0,  0),
                    (204, 0,  0),
                    (76, 153, 0),
                    (204, 204, 0),
                    (51, 51, 255),
                    (204, 0, 204),
                    (0, 255, 255),
                    (51, 255, 255),
                    (102, 51, 0),
                    (255, 0, 0),
                    (102, 204, 0),
                    (255, 255, 0),
                    (0, 0, 153),
                    (0, 0, 204),
                    (255, 51, 153), 
                    (0, 204, 204),
                    (0, 51, 0),
                    (255, 153, 51),
                    (0, 204, 0)], 
                    dtype=np.uint8)

    def __init__(self, sessionOptions = None):
        """
        Initializes face parser.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "face_unet_256.onnx")
        self.session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.input_name = self.session.get_inputs()[0].name

    def Forward(self, image, rectangles = None):
        """
        Returns face recognition results.
        Args:
            image: Bitmap
            rectanges: Rectangles

        Returns:
            Array
        """

        if rectangles is None:
            return self.__Forward(image)

        else:
            outputs = []

            for rectangle in rectangles:
                cropped = Crop(image, rectangle)
                points = self.__Forward(cropped)

                for i in range(len(points)):
                    points[i] += (rectangle[0], rectangle[1])

                outputs.append(points)

            return outputs

    def __Forward(self, image):
        """
        Returns face recognition results.
        Args:
            image: Bitmap

        Returns:
            Array
        """
        h, w, _ = image.shape
        size = (256, 256)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Resize(img, size)
        img = img / 255.0
        img_mean = np.array([0.5, 0.5, 0.5])
        img = (img - img_mean) / img_mean
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # post-processing
        confidences = self.session.run(None, {self.input_name: img})[0][0]
        s = confidences.shape[0]
        maximum = np.max(confidences)
        minimum = np.min(confidences)
        probabilities = (confidences - minimum) / (maximum - minimum)
        
        return probabilities

    def ToBitmap(self, masks):
        """
        Returns bitmap from masks.
        Args:
            masks: Masks

        Returns:
            image: Bitmap
        """
        s, h, w = masks.shape
        image = np.zeros((h, w, 3))

        for y in range(h):
            for x in range(w):
                maximum = -2147483648
                index = 0

                for i in range(s):
                    if (masks[i, y, x] > maximum):
                        maximum = masks[i, y, x]
                        index = i
                
                color = self.Labels[index]
                image[y, x] = color[::-1]

        return image