import cv2 as cv
import numpy as np
import math

# def line_x_at_y(x1, y1, x2, y2, y):
#     if y2 == y1: # if line is perfectly horizontal
#         return (x1 + x2) / 2.0 # use midpoint of x
#     # consider parameter t
#     # x(t) = x1 + t(x2 - x1)
#     # y(t) = y1 + t(y2 - y1)
#     # use y to solve for parameter t
#     t = (y - y1) / (y2 - y1)
#     # use parameter t to solve for x
#     return x1 + t * (x2 - x1)

# def is_coincident(x1, y1, x2, y2, threshold=10.0):
#     return abs(x2 - x1) < threshold and abs(y2 - y1) < threshold
#
# def is_corner(line1, line2, threshold=10.0):
#     x1, y1, x2, y2 = line1
#     x3, y3, x4, y4 = line2
#     if is_coincident(x1, y1, x3, y3, threshold): return True
#     if is_coincident(x1, y1, x4, y4, threshold): return True
#     if is_coincident(x2, y2, x3, y3, threshold): return True
#     if is_coincident(x2, y2, x4, y4, threshold): return True
#     return False

def is_perpendicular(line1, line2, angle_tolerance_deg=15.0):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)
    if mag1 == 0 or mag2 == 0:
        return False

    # cos(theta) = (v1·v2)/(|v1||v2|)
    dot_product = dx1 * dx2 + dy1 * dy2
    cos_theta = dot_product / (mag1 * mag2)

    # Clamp cos_theta [-1.0, 1.0] to use inverse cos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))

    return abs(theta - 90.0) <= angle_tolerance_deg

def detect_runway_edges(frame):
    # Preprocess
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0) # 0 means OpenCV auto picks sigma

    # Edge detection on whole image
    # lower and upper thresholds for hysteresis procedure
    # if greater than upper threshold than it is definitely an edge
    # if lower than lower threshold it is not an edge and discarded
    # if in between thresholds then it is weak edge and only kept if connected to strong edge
    edges = cv.Canny(blur, 50, 150)
    # debug
    cv.imshow("Edges", edges)

    # Hough line detection
    lines = cv.HoughLinesP(
        edges,
        rho=1, # consider distances in steps of 1 pixel
        theta=np.pi/180, # 1 degree steps
        threshold=100, # num of votes
        minLineLength=100, # pixels
        maxLineGap=20, # gap between two collinear segments
    )
    if lines is None:
        return None, frame

    output_frame = frame.copy()
    # separate lines into vertical and horizontal
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # find vertical lines using check: |dy|/|dx| > slope
        # use 3 for strict vertical, lower to allow more slant
        slope = 1.25
        if abs(y2 - y1) > abs(x2 -x1) * slope:
            # line is vertical
            vertical_lines.append((x1, y1, x2, y2))
            # draw it blue
            cv.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            # line is horizontal
            horizontal_lines.append((x1, y1, x2, y2))
            # draw it red
            cv.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # sort horizontal_lines using average of y1 and y2
    horizontal_lines = sorted(horizontal_lines, key=lambda L: (L[1] + L[3]) / 2.0, reverse=True)

    # loop through horizontal lines bottom up until at least two connected edges are found
    possible_lines = []
    base_line = None
    for horizontal_line in horizontal_lines:
        connected_lines = [] # reset
        for line in lines: # loop through lines instead?
            line = tuple(line[0])
            if line == horizontal_line:
                continue
            if is_perpendicular(horizontal_line, line):
                connected_lines.append(line)
        if len(connected_lines) >= 2:
            possible_lines = connected_lines
            base_line = horizontal_line
            break

    if len(possible_lines) < 2 or base_line is None:
        return None, frame

    # Take leftmost and rightmost as runway edges
    # sort using average of x1 and x2
    possible_lines = sorted(possible_lines, key=lambda L: (L[0] + L[2]) / 2.0)
    left = possible_lines[0]
    right = possible_lines[-1]

    # Draw runway edges green
    cv.line(output_frame, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 2)
    cv.line(output_frame, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 2)

    # Draw base_line blue
    hx1, hy1, hx2, hy2 = base_line
    cv.line(output_frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)

    return (left, right, base_line), output_frame