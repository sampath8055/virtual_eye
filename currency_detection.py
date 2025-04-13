import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("currency_detection.h5")
class_names = ['1Hundrednote', '2Hundrednote', '5Hundrednote', '2Thousandnote', 'Fiftynote', 'Tennote', 'TwentyNote']

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # shape becomes (1, 224, 224, 3)
    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 0), 2)

    roi = frame[100:324, 100:324] 
    input_img = preprocess_image(roi)
    pred = model.predict(input_img)
    predicted_class = class_names[np.argmax(pred)]

    cv2.putText(frame, f"Rs. {predicted_class}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Currency Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
