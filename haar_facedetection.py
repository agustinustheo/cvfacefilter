import os
import cv2
import numpy as np

# Define paths
base_dir = os.path.dirname(__file__)
face_cascade_path = os.path.join(base_dir + 'model/haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(base_dir + 'model/haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(face_cascade_path)

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

cap = cv2.VideoCapture('resources/pellek.mp4')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()