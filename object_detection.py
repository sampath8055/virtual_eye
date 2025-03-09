import cv2
import logging
from ultralytics import YOLO

# Disable YOLO logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the webcam (Change to 0 if needed)
cap = cv2.VideoCapture(0)

show_detections = False  # Flag to control when to print detections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF  # Non-blocking key detection

    if key == ord(" "):  # Spacebar toggles detection display
        show_detections = True

    if show_detections:
        print("\nObjects in Frame:")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Class index
                confidence = float(box.conf[0])  # Confidence score
                object_name = model.names[class_id]  # Object label
                
                print(f"- {object_name} ({confidence:.2f} confidence)")
        
        show_detections = False  # Reset flag after printing

    # Display the original frame without overlays
    frame_annotated = results[0].plot()
    cv2.imshow("Object Detection", frame_annotated)

    # Break loop if 'q' is pressed
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
