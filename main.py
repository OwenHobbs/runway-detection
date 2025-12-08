import os
import sys
import cv2 as cv
import runway
import edge_detection
from ultralytics import YOLO

# Load model
if (not os.path.exists("runway_model.pt")):
    print("ERROR: Model not found.")
    sys.exit(0)
model = YOLO("runway_model.pt", task='detect')

# initialize with external webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
resW, resH = 1280, 720
cap.set(3, resW) # width
cap.set(4, resH) # height

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

        # Draw runway bounding box
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Calculate lateral deviation
        lat_err_rw, lat_err_deg = runway.compute_lateral_error(xmin, xmax, w, hfov_deg=70)
        # Calculate vertical deviation
        vert_err = runway.compute_vertical_error(ymax, h, ideal_base_y_frac=0.70)
    else:
        # No runway detected
        cv.putText(frame, f"Runway not detected",
                (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display lateral deviation
    cv.putText(frame, f"Lateral error: {lat_err_rw:+.2f} RW widths, {lat_err_deg:+.1f} deg",
                (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display vertical deviation
    cv.putText(frame, f"Vertical error: {vert_err:+.2f}",
               (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow("Visual Runway Alignment Assist", frame)
    if cv.waitKey(1) == ord('q'):  # q to quit
        break

cap.release()
cv.destroyAllWindows()
