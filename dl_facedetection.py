from imutils.video import VideoStream
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import youtube_dl
import imutils
import time
import cv2
import os

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model/weights.caffemodel')
video_url = 'https://www.youtube.com/watch?v=kfchvCyHmsc'

ydl_opts = {}

# create youtube-dl object
ydl = youtube_dl.YoutubeDL(ydl_opts)

# set video url, extract video information
info_dict = ydl.extract_info(video_url, download=False)

# get video formats available
formats = info_dict.get('formats',None)
duration = info_dict.get('duration',None)

flag_for_testing = 0

for f in formats:
	url = f.get('url',None)
	if flag_for_testing == 1:
		break

	# I want the lowest resolution, so I set resolution as 144p
	if f.get('format_note',None) == '144p':
		
		#get the video url
		url = f.get('url',None)

		# load our serialized model from disk
		print("[INFO] loading model...")
		net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

		# initialize the video stream and allow the cammera sensor to warmup
		print("[INFO] starting video stream...")
		vs = cv2.VideoCapture(url)

		fps = vs.get(cv2.CAP_PROP_FPS)
		frame_count = 0
		timestamp = []
		start = 0
		# loop over the frames from the video stream
		while True:
			frame_count = frame_count + 1
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			ret, frame = vs.read()
			
			# check if frame is empty
			if not ret:
				break
			
			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
				(300, 300), (104.0, 177.0, 123.0))
		
			# pass the blob through the network and obtain the detections and
			# predictions
			net.setInput(blob)
			detections = net.forward()

			sum_confidence = 0
			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence < 0.5:
					continue

				sum_confidence = sum_confidence + confidence
				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
		
				# draw the bounding box of the face along with the associated
				# probability
				text = "{:.2f}%".format(confidence * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				# put the percentage text
				# cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			if sum_confidence < 0.5:
				end = float(frame_count)/fps
				# if face detection confidence is below 50% and the face detected index is not 0 then append face detected duration to timestamp
				if start != 0:
					timestamp.append( ({'start_time': start * duration} , {'end_time': end * duration}) )
					start = 0
			else:
				# if face detection confidence is above 50% and the face detected index is 0 then set new face detected index
				if start == 0:
					start = float(frame_count)/fps
				if frame_count == fps:
					end = float(frame_count)/fps
					timestamp.append( ({'start_time': start * duration} , {'end_time': end * duration}) )
					break

				model = tf.keras.models.load_model('model/cnn_face_expression.model')
				detected_face = frame[int(startY):int(startY+endY), int(startX):int(startX+endX)] #crop detected face
				detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
				detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				
				img_pixels /= 255
				
				predictions = model.predict(img_pixels)
				
				#find max indexed array
				max_index = np.argmax(predictions[0])
				
				emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
				emotion = emotions[max_index]
				
				cv2.putText(frame, emotion, (int(startX), int(startY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			flag_for_testing = 1

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

vs.release()
# do a bit of cleanup
cv2.destroyAllWindows()
for i in timestamp:
	print(i)