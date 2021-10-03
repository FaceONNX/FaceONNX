import math

def Add(points, point):
    """
    Returns processed points.
    Args:
        points: Points
        support: Point

    Returns:
        Points
    """
    output = []
    length = len(points)

    for i in range(length):
        current = points[i]
        output.append((current[0] + point[0], current[1] + point[1])

    return output

def Sub(points, point):
    """
    Returns processed points.
    Args:
        points: Points
        support: Point

    Returns:
        Points
    """
    output = []
    length = len(points)

    for i in range(length):
        current = points[i]
        output.append((current[0] - point[0], current[1] - point[1])

    return output

def GetAngle(left, right, support):
    """
    Return angle of the three points.
    Args:
        left: Left point
        right: Right point
        support: Supported point

    Returns:
        Angle
    """
    kk = 1 if (left[1] > right[1]) else -1

    x1 = left[0] - support[0]
    y1 = left[1] - support[1]

    x2 = right[0] - left[0]
    y2 = right[1] - left[1]

    cos = (x1 * x2 + y1 * y2) / math.sqrt(x1 * x1 + y1 * y1) / math.sqrt(x2 * x2 + y2 * y2)
    return kk * (180.0 - math.acos(cos) * 57.3)

def GetSupportPoint(left, right):
    """
    Returns supported point.
    Args:
        left: Left point
        right: Right point

    Returns:
        Point
    """
    return (right[0], left[1])

def GetMeanPoint(points):
    """
    Returns mean point.
    Args:
        points: Points

    Returns:
        Point
    """
    point = (0, 0)
    length = len(point)

    for i in range(length):
        point += points[i]

    point[0] /= length
    point[1] /= length

    return point

def GetRectangle(points):
    """
    Returns rectangle from face points.
    Args:
        points: Points

    Returns:
        Rectangle
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
    """
    Returns right eye points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(6):
        eye.append(points[i + 42])

    return eye

def GetLeftEye(points):
    """
    Returns left eye points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(6):
        eye.append(points[i + 36])

    return eye

def GetMouth(points):
    """
    Returns mouth points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(17):
        eye.append(points[i + 48])

    return eye

def GetFace(points):
    """
    Returns face points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(17):
        eye.append(points[i])

    return eye

def GetLeftBrow(points):
    """
    Returns left brow points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(5):
        eye.append(points[i + 17])

    return eye

def GetRightBrow(points):
    """
    Returns right brow points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(5):
        eye.append(points[i + 22])

    return eye

def GetNose(points):
    """
    Returns nose points.
    Args:
        points: Points

    Returns:
        Points
    """
    eye = []

    for i in range(9):
        eye.append(points[i + 27])

    return eye