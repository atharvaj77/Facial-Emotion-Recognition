import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('emotion_detector.h5')

rect_size = 4
cam = cv2.VideoCapture(0)

cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    (r_val, im) = cam.read()  # r_val = True or False
    im = cv2.flip(im, 1, 1)  # 0 is rotate about x axis, 1 is rotate around y axis

    resized_total_img = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = cascade_classifier.detectMultiScale(resized_total_img)

    for f in faces:
        (x, y, w, h) = [j * rect_size for j in f]

        face_img = im[y:y + h, x:x + w]  # Cutting image from background
        face_resized = cv2.resize(face_img, (150, 150))  # Resizing image to 150 , 150
        normalized = face_resized / 255.0  # Dividing every pixel by 255 to reduce size
        reshaped = np.reshape(normalized,
                              (1, 150, 150, 3))  # format = (Number of images, Width, Height, Color Channels)
        reshaped = np.vstack([reshaped])  # Stack faces one upon other
        result = model.predict(reshaped)  # Main function to get values (p(un),p(masked))

        label = np.argmax(result, axis=1)[0]  # Sort index of max value

        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 255, 0), 1)
        cv2.putText(im, str(result), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('EMOTIONS DETECTION REAL TIME', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cam.release()

cv2.destroyAllWindows()
