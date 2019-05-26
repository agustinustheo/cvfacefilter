import os
import cv2
import numpy as np
import youtube_dl

if __name__ == '__main__':

    # Define paths
    base_dir = os.path.dirname(__file__)
    face_cascade_path = os.path.join(base_dir + 'model/haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(base_dir + 'model/haarcascade_eye.xml')
    video_url = 'https://www.youtube.com/watch?v=d-YFLAKtcKI'

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    ydl_opts = {}

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)

    # get video formats available
    formats = info_dict.get('formats',None)

    for f in formats:
        
        # I want the lowest resolution, so I set resolution as 144p
        if f.get('format_note',None) == '360p':
            
            #get the video url
            url = f.get('url',None)

            # open url with opencv
            cap = cv2.VideoCapture(url)

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
                faces = face_cascade.detectMultiScale(gray)
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                # display frame
                cv2.imshow('frame', frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            # release VideoCapture
            cap.release()

    cv2.destroyAllWindows()