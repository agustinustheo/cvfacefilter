import os
import sys
import cv2
import numpy as np

if __name__ == '__main__':

    # Define paths
    base_dir = os.path.dirname(__file__)
    face_cascade_path_alt = os.path.join(base_dir + 'model/haarcascade_frontalface_alt2.xml')
    eye_cascade_path = os.path.join(base_dir + 'model/haarcascade_eye.xml')
    nose_cascade_path = os.path.join(base_dir + 'model/haarcascade_mcs_nose.xml')
    mouth_cascade_path = os.path.join(base_dir + 'model/haarcascade_mcs_mouth.xml')

    face_cascade = cv2.CascadeClassifier(face_cascade_path_alt)
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

    # open url with opencv
    frame = cv2.imread('resources/lena.png')
    img = cv2.imread('filter/dog_ears.png', cv2.IMREAD_UNCHANGED)
    img_height, img_width, _ = img.shape
    nose_filter = cv2.imread('filter/dog_nose.png', cv2.IMREAD_UNCHANGED)
    nose_filter_height, nose_filter_width, _ = nose_filter.shape
    mouth_filter = cv2.imread('filter/tongue.png', cv2.IMREAD_UNCHANGED)
    mouth_filter_height, mouth_filter_width, _ = mouth_filter.shape

    filter_layer = np.zeros((frame.shape[0], frame.shape[1], 4))
    nose_filter_layer = np.zeros((frame.shape[0], frame.shape[1], 4))
    mouth_filter_layer = np.zeros((frame.shape[0], frame.shape[1], 4))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(10,10)
    )

    # new_width, new_height = 0, 0
    for (x,y,w,h) in faces:
        if img_width >= img_height:
            new_height = int(h + 40)
            new_width = int((img_width / img_height * new_height))
            resize_img = cv2.resize(img, (new_width, new_height))
        else:
            new_width = int(w + 40)
            new_height = int((img_height / img_width * new_width))
            resize_img = cv2.resize(img, (new_width, new_height))

        x1 = int(x - ((new_width - w) / 2))
        y1 = int(y - ((new_height - h) / 2))

        x1_img = 0
        y1_img = 0
        if y1 <= 0:
            y1 = 0
            y1_img = y1 * -1
        if x1 <= 0:
            x1 = 0
            x1_img = x1 * -1

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            filter_layer[y1:y1+new_height-y1_img, x1:x1+new_width-x1_img] = resize_img[y1_img:new_height, x1_img:new_width]
            res = frame[:] #copy the first layer into the resulting image

            cnd = filter_layer[:, :, 3] > 0
            res[cnd] = filter_layer[cnd]
        except Exception as e:
            print(e)
            break
        
        # draw rectangle around face 
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

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
        mouth = mouth_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(5,5)
        )
        
        nose_to_face_x_center = sys.maxsize
        nose_to_face_y_center = sys.maxsize
        nose_x, nose_y, nose_w, nose_h = 0, 0, 0, 0
        for (ex, ey, ew, eh) in nose:
            x_nose_to_center = abs((ex + (ew / 2)) - (x + (w / 2)))
            y_nose_to_center = abs((ey + (eh / 2)) - (y + (h / 2)))
            if x_nose_to_center < nose_to_face_x_center and y_nose_to_center < nose_to_face_y_center:
                    nose_to_face_x_center = x_nose_to_center
                    nose_to_face_y_center = y_nose_to_center

                    nose_x, nose_y, nose_w, nose_h = ex, ey, ew, eh

        # draw rectangle around nose
        # cv2.rectangle(roi_color, (nose_x, nose_y), (nose_x+nose_w, nose_y+nose_h), (0, 0, 255), 2)

        if nose_w > 0 and nose_h > 0:
            if nose_filter_width >= nose_filter_height:
                new_nose_height = int(nose_h)
                new_nose_width = int((nose_filter_width / nose_filter_height * new_nose_height))
                resize_nose_img = cv2.resize(nose_filter, (new_nose_width, new_nose_height))
            else:
                new_nose_width = int(nose_w)
                new_nose_height = int((nose_filter_height / nose_filter_width * new_nose_width))
                resize_nose_img = cv2.resize(nose_filter, (new_nose_width, new_nose_height))

            nose_x1 = int(nose_x - ((new_nose_width - nose_w) / 2))
            nose_y1 = int(nose_y - ((new_nose_height - nose_h) / 2))

            nose_x1_img = 0
            nose_y1_img = 0
            if nose_y1 <= 0:
                nose_y1_img = nose_y1 * -1
                nose_y1 = 0
            if nose_x1 <= 0:
                nose_x1_img = nose_x1 * -1
                nose_x1 = 0

            try:
                nose_in_face_x1 = int(x)
                nose_in_face_y1 = int(y)
                nose_filter_layer[nose_in_face_y1+nose_y1:nose_in_face_y1+nose_y1+new_nose_height-nose_y1_img, nose_in_face_x1+nose_x1:nose_in_face_x1+nose_x1+new_nose_width-nose_x1_img] = resize_nose_img[nose_y1_img:new_nose_height, nose_x1_img:new_nose_width]
                nose_cnd = nose_filter_layer[:, :, 3] > 0
                res[nose_cnd] = nose_filter_layer[nose_cnd]
            except Exception as e:
                print(e)

        mouth_to_face_x_center = sys.maxsize
        mouth_x, mouth_y, mouth_w, mouth_h = 0, 0, 0, 0
        for (ex, ey, ew, eh) in mouth:
            x_mouth_to_center = abs((ex + (ew / 2)) - (x + (w / 2)))
            if x_mouth_to_center < mouth_to_face_x_center and nose_y - ey < 0 and ey > nose_y + nose_h:
                    mouth_to_face_x_center = x_mouth_to_center
                    mouth_x, mouth_y, mouth_w, mouth_h = ex, ey, ew, eh

        # draw rectangle around mouth
        # cv2.rectangle(roi_color, (mouth_x, mouth_y), (mouth_x+mouth_w, mouth_y+mouth_h), (0, 0, 255), 2)

        if mouth_w > 0 and mouth_h > 0:
            if mouth_filter_width >= mouth_filter_height:
                new_mouth_height = int(mouth_h)
                new_mouth_width = int((mouth_filter_width / mouth_filter_height * new_mouth_height))
                resize_mouth_img = cv2.resize(mouth_filter, (new_mouth_width, new_mouth_height))
            else:
                new_mouth_width = int(mouth_w)
                new_mouth_height = int((mouth_filter_height / mouth_filter_width * new_mouth_width))
                resize_mouth_img = cv2.resize(mouth_filter, (new_mouth_width, new_mouth_height))

            mouth_x1 = int(mouth_x - ((new_mouth_width - mouth_w) / 2))
            mouth_y1 = int(mouth_y - ((new_mouth_height - mouth_h) / 2))

            mouth_x1_img = 0
            mouth_y1_img = 0
            if mouth_y1 <= 0:
                mouth_y1 = 0
                mouth_y1_img = mouth_y1 * -1
            if mouth_x1 <= 0:
                mouth_x1 = 0
                mouth_x1_img = mouth_x1 * -1

            try:
                mouth_in_face_x1 = int(x)
                mouth_in_face_y1 = int(y + (mouth_h / 2))
                mouth_filter_layer[mouth_in_face_y1+mouth_y1:mouth_in_face_y1+mouth_y1+new_mouth_height-mouth_y1_img, mouth_in_face_x1+mouth_x1:mouth_in_face_x1+mouth_x1+new_mouth_width-mouth_x1_img] = resize_mouth_img[mouth_y1_img:new_mouth_height, mouth_x1_img:new_mouth_width]
                mouth_cnd = mouth_filter_layer[:, :, 3] > 0
                res[mouth_cnd] = mouth_filter_layer[mouth_cnd]
            except Exception as e:
                print(e)

    # display frame
    cv2.imshow('frame', res)
    cv2.waitKey(0)