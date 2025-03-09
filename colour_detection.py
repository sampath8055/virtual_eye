import cv2
import numpy as np

color_ranges = {
    "Red": ([0, 50, 50], [10, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Green": ([40, 40, 40], [80, 255, 255]),
    "Blue": ([90, 50, 50], [130, 255, 255]),
    "Purple": ([130, 50, 50], [160, 255, 255]),
    "Orange": ([10, 100, 100], [20, 255, 255]),
    "White": ([0, 0, 200], [180, 40, 255]),
    "Black": ([0, 0, 0], [180, 255, 50])
}

def get_color_name(hsv_value):
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        if np.all(lower <= hsv_value) and np.all(hsv_value <= upper):
            return color
    return "Unknown"


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2  

    box_size = 100
    x1, y1 = cx - box_size, cy - box_size
    x2, y2 = cx + box_size, cy + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    box_hsv = hsv_frame[y1:y2, x1:x2]
    avg_hsv = np.mean(box_hsv, axis=(0, 1)).astype(int)  # Average HSV value
    detected_color = get_color_name(avg_hsv)

    cv2.putText(frame, "Place object in the box & Press Space", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow("Color Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):  # Press Space to detect color
        print(f"Detected Color: {detected_color}")
        print(f"HSV Value: {avg_hsv}")

    if key == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
