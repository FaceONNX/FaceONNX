import numpy as np
import math

def GetAngle(left, right, support):
    """[summary]

    Args:
        left ([type]): [description]
        right ([type]): [description]
        support ([type]): [description]

    Returns:
        [type]: [description]
    """
    kk = 1 if (left[1] > right[1]) else -1

    x1 = left[0] - support[0]
    y1 = left[1] - support[1]

    x2 = right[0] - left[0]
    y2 = right[1] - left[1]

    cos = (x1 * x2 + y1 * y2) / math.sqrt(x1 * x1 + y1 * y1) / math.sqrt(x2 * x2 + y2 * y2)
    return kk * (180.0 - math.acos(cos) * 57.3)

def GetSupportPoint(left, right):
    """[summary]

    Args:
        left ([type]): [description]
        right ([type]): [description]

    Returns:
        [type]: [description]
    """
    return right[0], left[1]

def GetMeanPoint(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    point = (0, 0)
    length = len(point)

    for i in range(length):
        point += points[i]

    point[0] /= length
    point[1] /= length

    return point

def GetRectangle(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    length = points.Length
    xmin = 2147483647
    ymin = 2147483647
    xmax = -2147483648
    ymax = -2147483648

    for i in range(length):
        x = points[i][0]
        y = points[i][1]

        if (x < xmin):
            xmin = x
        if (y < ymin):
            ymin = y
        if (x > xmax):
            xmax = x
        if (y > ymax):
            ymax = y
        
        return [xmin, ymin, xmax - xmin, ymax - ymin]

def GetRightEye(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(6):
        eye.append(points[i + 42])

    return eye

def GetLeftEye(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(6):
        eye.append(points[i + 36])

    return eye

def GetMouth(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(17):
        eye.append(points[i + 48])

    return eye

def GetFace(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(17):
        eye.append(points[i])

    return eye

def GetLeftBrow(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(5):
        eye.append(points[i + 17])

    return eye

def GetRightBrow(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(5):
        eye.append(points[i + 22])

    return eye

def GetNose(points):
    """[summary]

    Args:
        points ([type]): [description]

    Returns:
        [type]: [description]
    """
    eye = []

    for i in range(9):
        eye.append(points[i + 27])

    return eye