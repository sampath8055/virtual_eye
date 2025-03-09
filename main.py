import cv2
import logging
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO("yolov8n.pt")

# Open the webcam (Change to 0 if needed)
cap = cv2.VideoCapture(0)

show_detections = False 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    key = cv2.waitKey(1) & 0xFF  

    if key == ord(" "):  
        show_detections = True

    if show_detections:
        print("\nObjects in Frame:")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  
                confidence = float(box.conf[0])  
                object_name = model.names[class_id]  # 
                
                print(f"- {object_name} ({confidence:.2f} confidence)")
        
        show_detections = False  

    # Break loop if 'q' is pressed
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
