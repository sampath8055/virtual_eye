import cv2
import logging
import numpy as np
import sys
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

show_detections = False  
detect_color = False  

def classify_color(hsv_value):
    h, s, v = hsv_value
    
    if v < 50:
        return "Black"
    elif v > 200 and s < 30:
        return "White"
    elif s < 50:
        return "Gray"
    
    if 0 <= h < 10 or 170 <= h <= 180:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 125:
        return "Blue"
    elif 125 <= h < 150:
        return "Purple"
    elif 150 <= h < 170:
        return "Pink"
    
    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    print("\nPress 's' to toggle object detection, 'c' for color detection, 'q' to quit:")
    user_input = sys.stdin.read(1).strip()  # Read a single key from input

    if user_input == "s":
        show_detections = not show_detections
        print("Toggled Object Detection:", "ON" if show_detections else "OFF")
    
    if user_input == "c":
        detect_color = not detect_color
        print("Toggled Color Detection:", "ON" if detect_color else "OFF")

    if show_detections:
        print("\nObjects in Frame:")
        result = results[0]
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls[0])  
                confidence = float(box.conf[0])  
                object_name = model.names[class_id]  
                
                print(f"- {object_name} ({confidence:.2f} confidence)")
        show_detections = False  

    if detect_color:
        result = results[0]
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                object_roi = frame[y1:y2, x1:x2]  
                if object_roi.size == 0:
                    continue  
                
                hsv_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2HSV)  # Convert to HSV
                avg_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)  # Get average HSV
                detected_color = classify_color(avg_hsv)
                print(f"Detected Color: {detected_color}")
        detect_color = False

    if user_input == "q":
        print("\nExiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
