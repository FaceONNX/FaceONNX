import onnxruntime
import numpy
import cv2
import os
from .rectangles import ToBox
from .imaging import Rotate, Resize, Crop
from .landmarks import GetMeanPoint, GetSupportPoint, GetAngle, GetLeftEye, GetRightEye
from .internal import Detector
from google_drive_downloader import GoogleDriveDownloader as g

# main
__this = os.path.dirname(__file__)
models_path = os.path.join(__this, "models")

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
        g.download_file_from_google_drive(file_id='1tB7Y5l5Jf2270IisgSZ3rbCQs0-pkNIQ', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name
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
        img_mean = numpy.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = numpy.transpose(img, [2, 0, 1])
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)
        confidences, boxes = self.__session.run(None, {self.__input_name: img})
        boxes, labels, probes = Detector(w, h, confidences, boxes, self.confidenceThreshold, self.nmsThreshold)

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
        g.download_file_from_google_drive(file_id='1T39Qh7FA-tNbDge6PE4gN-8Ajgnmozum', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name

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
        img_mean = numpy.array([104, 117, 123])
        img = img - img_mean
        img = numpy.transpose(img, [2, 0, 1])
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)

        return self.__session.run(None, {self.__input_name: img})[0][0]

class FaceBeautyClassifier:

    def __init__(self, sessionOptions = None):
        """
        Initializes face beauty classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "beauty_resnet18.onnx")
        g.download_file_from_google_drive(file_id='1Eqr3KXXEFI2vhFAggknmNdmKtO0Ap_0C', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name

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
        img_mean = numpy.array([104, 117, 123])
        img = (img - img_mean) / 255.0
        img = numpy.transpose(img, [2, 0, 1])
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)

        return round(2.0 * sum(self.__session.run(None, {self.__input_name: img})[0][0]), 1)

class FaceEmbedder:

    def __init__(self, sessionOptions = None):
        """
        Initializes face embedder.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "recognition_resnet27.onnx")
        g.download_file_from_google_drive(file_id='1ijbMt1LETLQc6GDGAtEJx8ggEGenyM7m', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name

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
        img_mean = numpy.array([127.5, 127.5, 127.5])
        img = (img - img_mean) / 128
        img = numpy.transpose(img, [2, 0, 1])
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)

        return self.__session.run(None, {self.__input_name: img})[0][0]

class FaceEmotionClassifier:

    """
    Returns the labels.
    Returns:
        Labels
    """
    Labels = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disguest", "Fear"]

    def __init__(self, sessionOptions = None):
        """
        Initializes face emotion classifier.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "emotion_cnn.onnx")
        g.download_file_from_google_drive(file_id='1Oqd-0klyn-loAnUyXdah4FN131YfDFcv', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name

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
        img = numpy.expand_dims(img, axis=0)
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)

        return numpy.exp(self.__session.run(None, {self.__input_name: img})[0][0])

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
        g.download_file_from_google_drive(file_id='1ZsqnXunyEgxaAx9WoX5uQv_T7RWvTbTz', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name

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
        img_mean = numpy.array([104, 117, 123])
        img = img - img_mean
        img = numpy.transpose(img, [2, 0, 1])
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)

        return self.__session.run(None, {self.__input_name: img})[0][0]

class FaceLandmarksExtractor:

    def __init__(self, sessionOptions = None):
        """
        Initializes face landmarks extractor.
        Args:
            sessionOptions: Session options.
        """
        onnx_path = os.path.join(models_path, "landmarks_68_pfld.onnx")
        g.download_file_from_google_drive(file_id='1qgM6ZqMyB60FYlzzxNDyUefifLS0lhag', dest_path=onnx_path)
        self.__session = onnxruntime.InferenceSession(onnx_path, sessionOptions)
        self.__input_name = self.__session.get_inputs()[0].name

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
        img = numpy.transpose(img, [2, 0, 1])
        img = numpy.expand_dims(img, axis=0)
        img = img.astype(numpy.float32)

        output = self.__session.run(None, {self.__input_name: img})[0][0]
        points = output.reshape(-1, 2) * (w, h)
        return points.astype(numpy.int32)

    @staticmethod
    def Align(image, points):
        """
        Returns aligned face.
        Args:
            image: Bitmap
            points: Points

        Returns:
            Bitmap
        """
        left = GetMeanPoint(GetLeftEye(points))
        right = GetMeanPoint(GetRightEye(points))
        point = GetSupportPoint(left, right)
        angle = GetAngle(left, right, point)
        return Rotate(image, angle)