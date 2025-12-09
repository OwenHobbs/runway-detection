import os
import sys
import cv2 as cv
import runway
from ultralytics import YOLO

# Load model
if (not os.path.exists("runway_model.pt")):
    print("ERROR: Model not found.")
    sys.exit(0)
model = YOLO("runway_model.pt", task='detect')

# initialize with external webcam
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
resW, resH = 1280, 720
cap.set(3, resW) # width
cap.set(4, resH) # height

lat_err_rw = 0
lat_err_deg = 0
vert_err_px = 0
vert_err_deg = 0
while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    # break if no frame captured (ret is false)
    if not ret:
        print("Cannot read camera")
        break

    # Resize frame to desired resolution
    frame = cv.resize(frame,(resW,resH))

    # Run model!
    results = model(frame, verbose=False)
    detections = results[0].boxes

    if len(detections) != 0:
        # sort detections by confidence
        detections = sorted(detections, key=lambda det: det.conf.item(), reverse=True)
        # Use highest-confidence detection
        runway_box = detections[0]
        xyxy_tensor = runway_box.xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
        h, w, _ = frame.shape

        # Calculate lateral deviation
        lat_err_rw, lat_err_deg = runway.compute_lateral_error(xmin, xmax, w, hfov_deg=70)

        # Calculate vertical deviation
        ref_y_frac = 0.70
        vert_err_px, vert_err_deg = runway.compute_vertical_error(ymax, h, ideal_base_y_frac=ref_y_frac, vfov_deg=40)

        is_aligned = (abs(lat_err_deg) < 1) and (abs(vert_err_deg) < 1)

        # Draw runway bounding box
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0) if is_aligned else (0, 0, 255) , 2)

        # Draw fixed line at midframe
        mid_x = int(w / 2.0)
        cv.line(frame, (mid_x, 0), (mid_x, h), (0, 255, 0) if is_aligned else (255, 0, 0), 1)
        # Draw runway centerline
        center_x = int((xmin + xmax) / 2.0)
        cv.line(frame, (center_x, 0), (center_x, h), (0, 255, 0) if is_aligned else (0, 0, 255), 1)

        # Draw fixed line at ref_y
        ref_y = int(ref_y_frac * h)
        cv.line(frame, (0, ref_y), (w, ref_y), (0, 255, 0) if is_aligned else (255, 0, 0), 1)
        # Draw runway threshold
        cv.line(frame, (0, ymax), (w, ymax), (0, 255, 0) if is_aligned else (0, 0, 255), 1)
    else:
        # No runway detected
        cv.putText(frame, f"Runway not detected",
                (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display lateral deviation
    cv.putText(frame, f"Lateral error: {lat_err_rw:+.2f} RW widths, {lat_err_deg:+.1f} deg",
                (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display vertical deviation
    cv.putText(frame, f"Vertical error: {vert_err_px:+.2f} px, {vert_err_deg:+.1f} deg",
               (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow("Visual Runway Alignment Assist", frame)
    if cv.waitKey(1) == ord('q'):  # q to quit
        break

cap.release()
cv.destroyAllWindows()
