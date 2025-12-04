import cv2 as cv
import numpy as np

def line_x_at_y(x1, y1, x2, y2, y):
    if y2 == y1:
        return (x1 + x2) / 2.0
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

def compute_lateral_error(left, right, frame_width, ref_y):
    x1L, y1L, x2L, y2L = left
    x1R, y1R, x2R, y2R = right

    xL = line_x_at_y(x1L, y1L, x2L, y2L, ref_y)
    xR = line_x_at_y(x1R, y1R, x2R, y2R, ref_y)

    centerline_x = 0.5 * (xL + xR)
    image_center_x = frame_width / 2.0

    half_width = 0.5 * (xR - xL)
    if half_width == 0:
        return 0.0

    lateral_error_norm = (image_center_x - centerline_x) / half_width
    return lateral_error_norm  # +1 = 1 runway-width right, -1 = left

def detect_runway_edges(frame):
    # Preprocess
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # Edge detection on whole image
    edges = cv.Canny(blur, 50, 150)

    # Hough line detection
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=20,
    )

    if lines is None:
        return None, frame

    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # find near vertical lines
        if abs(x2 - x1) < abs(y2 - y1) * 0.5:
            vertical_lines.append((x1, y1, x2, y2))

    if len(vertical_lines) < 2:
        return None, frame

    # Take leftmost and rightmost as runway edges
    vertical_lines = sorted(vertical_lines, key=lambda L: (L[0] + L[2]) / 2.0)
    left = vertical_lines[0]
    right = vertical_lines[-1]

    output_frame = frame.copy()
    # Draw runway edges
    cv.line(output_frame, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 2)
    cv.line(output_frame, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 2)

    return (left, right), output_frame

# initialize with external webcam
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    # break if no frame captured (ret is false)
    if not ret:
        print("Cannot read camera")
        break

    # display_frame is copy of frame but with runway edges highlighted
    runway_edges, display_frame = detect_runway_edges(frame)
    h, w, _ = display_frame.shape

    if runway_edges is not None:
        left, right = runway_edges

        # Calculate lateral deviation at ref_y
        ref_y = int(h * 0.5)
        lat_err = compute_lateral_error(left, right, w, ref_y)

        # Display reference line
        cv.line(display_frame, (0, ref_y), (w, ref_y), (255, 0, 0), 1)
        # Display lateral deviation
        cv.putText(display_frame, f"Lateral error: {lat_err:+.2f} RW widths",
                    (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow("Visual Runway Alignment Assist", display_frame)
    if cv.waitKey(1) == ord('q'):  # q to quit
        break

cap.release()
cv.destroyAllWindows()
