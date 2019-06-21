import os
import cv2
import numpy as np

if __name__ == '__main__':

    # Define paths
    base_dir = os.path.dirname(__file__)
    face_cascade_path = os.path.join(base_dir + 'model/haarcascade_frontalface_default.xml')
    face_cascade_path_tree = os.path.join(base_dir + 'model/haarcascade_frontalface_alt_tree.xml')
    face_cascade_path_alt = os.path.join(base_dir + 'model/haarcascade_frontalface_alt2.xml')
    face_cascade_path_alt2 = os.path.join(base_dir + 'model/haarcascade_frontalface_alt.xml')
    # eye_cascade_path = os.path.join(base_dir + 'model/haarcascade_eye.xml')
    eye_cascade_path = os.path.join(base_dir + 'model/haarcascade_eye.xml')
    nose_cascade_path = os.path.join(base_dir + 'model/Nariz.xml')

    face_cascade = cv2.CascadeClassifier(face_cascade_path_alt)
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # open url with opencv
    cap = cv2.VideoCapture('resources/pellek.mp4')
    img = cv2.imread('resources/jackman.jpg')
    img_height, img_width, _ = img.shape

    # check if url was opened
    if not cap.isOpened():
        print('video not opened')
        exit(-1)

    while True:
        # read frame
        ret, frame = cap.read()

        # check if frame is empty
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(10,10)
        )
        for (x,y,w,h) in faces:
            if w > h:
                new_width = w
                new_height = img_height / img_width * new_width
                resize_img = cv2.resize(img, (new_width, new_height))
            else h > w:
                new_height = h
                new_width = img_width / img_height * new_height
                resize_img = cv2.resize(img, (new_width, new_height))

            mid_face_width = int(w/2)
            mid_face_height = int(h/2)

            resize_img_height, resize_img_width, _ = resize_img.shape

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(5,5)
            )
            nose = nose_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(5,5)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # for (ex, ey, ew, eh) in nose:
            #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

        # display frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # release VideoCapture
    cap.release()
    cv2.destroyAllWindows()