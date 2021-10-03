import numpy
import cv2

def Crop(image, rectangle):
    """
    Returns cropped image.
    Args:
        image: Bitmap
        rectangle: Rectangle

    Returns:
        Bitmap
    """
    h, w, _ = image.shape

    x0 = max(min(w, rectangle[0]), 0)
    x1 = max(min(w, rectangle[2]), 0)
    y0 = max(min(h, rectangle[1]), 0)
    y1 = max(min(h, rectangle[3]), 0)

    num = image[y0:y1, x0:x1]
    return num

def Resize(image, size):
    """
    Returns resized image.
    Args:
        image: Bitmap
        size: Size

    Returns:
        Bitmap
    """
    return cv2.resize(image, size)

def Rotate(image, angle):
    """
    Returns rotated image.
    Args:
        image: Bitmap
        angle: Angle

    Returns:
        Bitmap
    """
    image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result