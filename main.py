import cv2 as cv
import runway
import edge_detection

# initialize with external webcam
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

lat_err_rw = 0
lat_err_deg = 0
vert_err = 0
while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    # break if no frame captured (ret is false)
    if not ret:
        print("Cannot read camera")
        break

    # display_frame is copy of frame but with runway edges highlighted
    runway_edges, display_frame = edge_detection.detect_runway_edges(frame)
    h, w, _ = display_frame.shape

    if runway_edges is not None:
        left, right, base_line = runway_edges

        # Calculate lateral deviation
        left_x = (left[0] + left[2]) / 2.0 # average x1 and x2
        right_x = (right[0] + right[2]) / 2.0  # average x1 and x2
        lat_err_rw, lat_err_deg = runway.compute_lateral_error(left_x, right_x, w, hfov_deg=70)

        # Calculate vertical deviation
        base_y = (base_line[1] + base_line[3]) / 2.0 # average y1 and y2
        vert_err = runway.compute_vertical_error(base_y, h, ideal_base_y_frac=0.70)

    # Display lateral deviation
    cv.putText(display_frame, f"Lateral error: {lat_err_rw:+.2f} RW widths, {lat_err_deg:+.1f} deg",
                (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display vertical deviation
    cv.putText(display_frame, f"Vertical error: {vert_err:+.2f}",
               (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow("Visual Runway Alignment Assist", display_frame)
    if cv.waitKey(1) == ord('q'):  # q to quit
        break

cap.release()
cv.destroyAllWindows()
