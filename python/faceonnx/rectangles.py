import numpy
import cv2

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
