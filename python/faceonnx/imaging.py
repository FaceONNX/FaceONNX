import numpy as np
import cv2

def Rotate(image, angle):
    """[summary]

    Args:
        image ([type]): [description]
        angle ([type]): [description]

    Returns:
        [type]: [description]
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def Resize(image, size):
    """[summary]

    Args:
        image ([type]): [description]
        size ([type]): [description]

    Returns:
        [type]: [description]
    """
    return cv2.resize(image, size)

def Crop(image, box):
    """[summary]

    Args:
        image ([type]): [description]
        box ([type]): [description]

    Returns:
        [type]: [description]
    """
    h, w, _ = image.shape

    x0 = max(min(w, box[0]), 0)
    x1 = max(min(w, box[2]), 0)
    y0 = max(min(h, box[1]), 0)
    y1 = max(min(h, box[3]), 0)

    num = image[y0:y1, x0:x1]
    return num

def Area(box):
    """[summary]

    Args:
        box ([type]): [description]

    Returns:
        [type]: [description]
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w * h

def Max(boxes):
    """[summary]

    Args:
        boxes ([type]): [description]

    Returns:
        [type]: [description]
    """
    index = 0
    area = -2147483648

    for i in range(len(boxes)):
        box = boxes[i]
        cur = Area(box)

        if (cur > area):
            area = cur
            index = i

    return boxes[index]

def Min(boxes):
    """[summary]

    Args:
        boxes ([type]): [description]

    Returns:
        [type]: [description]
    """
    index = 0
    area = 2147483647

    for i in range(len(boxes)):
        box = boxes[i]
        cur = Area(box)

        if (cur < area):
            area = cur
            index = i

    return boxes[index]

def ToBox(box):
    """[summary]

    Args:
        box ([type]): [description]

    Returns:
        [type]: [description]
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    m = max(width, height)
    dx = int((m - width)/2)
    dy = int((m - height)/2)
    
    return [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
