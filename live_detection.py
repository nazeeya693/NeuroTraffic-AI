from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic1.mp4")

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# Check if video opens
if not cap.isOpened():
    print("ERROR: Cannot open video.")
    exit()

# Vehicle classes
vehicle_classes = ["car", "bus", "truck", "motorbike"]

# Main loop
while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Vehicle count
    vehicle_count = 0

    # Emergency detection
    emergency_detected = False

    # Loop through detections
    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            class_name = model.names[cls]

            # Count vehicles
            if class_name in vehicle_classes:
                vehicle_count += 1

            # Emergency concept
            # Using truck as demo emergency vehicle
            if class_name == "truck":
                emergency_detected = True

    # Traffic density logic
    if vehicle_count < 10:
        density = "LOW"

    elif vehicle_count < 20:
        density = "MEDIUM"

    else:
        density = "HIGH"

    # Draw detections
    annotated_frame = results[0].plot()

    # Vehicle count text
    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Traffic density text
    cv2.putText(
        annotated_frame,
        f"Traffic Density: {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Emergency alert
    if emergency_detected:

        cv2.putText(
            annotated_frame,
            "EMERGENCY VEHICLE DETECTED!",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        cv2.putText(
            annotated_frame,
            "INCREASE GREEN SIGNAL TIME",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

    # Show video
    cv2.imshow("NeuroTraffic AI", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()

cv2.destroyAllWindows()