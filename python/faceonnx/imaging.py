import numpy as np
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
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def Area(rectangle):
    """
    Returns rectangle area.
    Args:
        rectangle: Rectangle

    Returns:
        Area
    """
    w = rectangle[2] - rectangle[0]
    h = rectangle[3] - rectangle[1]
    return w * h

def Max(rectangles):
    """
    Returns the maximum rectangle.
    Args:
        rectangles: Rectangles

    Returns:
        Rectangle
    """
    index = 0
    area = -2147483648

    for i in range(len(rectangles)):
        box = rectangles[i]
        cur = Area(box)

        if (cur > area):
            area = cur
            index = i

    return rectangles[index]

def Min(rectangles):
    """
    Returns the minimum rectangle.
    Args:
        rectangles: Rectangles

    Returns:
        Rectangle
    """
    index = 0
    area = 2147483647

    for i in range(len(rectangles)):
        box = rectangles[i]
        cur = Area(box)

        if (cur < area):
            area = cur
            index = i

    return rectangles[index]

def ToBox(rectangle):
    """
    Returns rectangle scaled to box.
    Args:
        rectangle: Rectangle

    Returns:
        Rectangle
    """
    width = rectangle[2] - rectangle[0]
    height = rectangle[3] - rectangle[1]
    m = max(width, height)
    dx = int((m - width)/2)
    dy = int((m - height)/2)
    
    return [rectangle[0] - dx, rectangle[1] - dy, rectangle[2] + dx, rectangle[3] + dy]
