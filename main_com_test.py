"""
pip install dlib
pip install face_recognition
pip install imutils
pip install opencv-contrib-python
pip install pyttxx3
pip install webcolors
"""
# import the necessary packages
import RPi.GPIO as GPIO
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pyttsx3 as pyttsx
import webcolors


GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)
mode = 1
encoding_file = "encodings_2.pickle"
cascade_file = "haarcascade_frontalface_default.xml"
confidence = 0.5
padding = 0.05
engine = pyttsx.init()
# set tts wpm
engine.setProperty('rate',180)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encoding_file, "rb").read())
detector = cv2.CascadeClassifier(cascade_file)
engine.say("loading encodings")
engine.runAndWait()

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
#vs = VideoStream(src=1).start()
engine.say("starting video stream")
engine.runAndWait()

try:
	# loop over frames from the video file stream
	while True:

		# grab the frame from the threaded video stream
		frame = vs.read()
		#cv2.imshow("Frame", frame)

		if GPIO.input(17):

			time.sleep(0.08)

			if GPIO.input(17):
				start_time = time.time()

				while GPIO.input(17):
					continue

				buttonTime = time.time() - start_time
				if buttonTime > 1:

					if mode == 1:
						mode = 2
						print("Colour Recognition Mode")
						engine.say("Colour Recognition Mode")
						engine.runAndWait()

					elif mode == 2:
						mode = 1
						print("Face Recognition Mode")
						engine.say("Face Recognition Mode")
						engine.runAndWait()

				else:

					if mode == 1:

						print("Picture Taken")

						frame = imutils.resize(frame, width=500)
						gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

						# detect faces in the grayscale frame
						rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
														  flags=cv2.CASCADE_SCALE_IMAGE)

						# OpenCV returns bounding box coordinates in (x, y, w, h) order
						# but we need them in (top, right, bottom, left) order, so we
						# need to do a bit of reordering
						boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

						encodings = face_recognition.face_encodings(rgb, boxes)
						names = []

						for encoding in encodings:

							# attempt to match each face in the input image to our known encodings
							matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
							name = "Unknown"

							# check to see if we have found a match
							if True in matches:

								# find the indexes of all matched faces
								matchedIdxs = [i for (i, b) in enumerate(matches) if b]

								# initialize dictionary to count face matches
								counts = {}

								# loop over the matched indexes and maintain a count for each recognized face face
								for i in matchedIdxs:
									name = data["names"][i]
									counts[name] = counts.get(name, 0) + 1

								# determine the recognized face with the largest number of votes
								name = max(counts, key=counts.get)

							# update the list of names
							print(name)
							engine.say(name.replace("_", " "))
							engine.runAndWait()
							names.append(name)

					elif mode == 2:

						frame = imutils.resize(frame, width=500)
						image_height = frame.shape[0]  # get image height
						image_width = frame.shape[1]  # get image width

						ymid = int(image_height / 2)
						xmid = int(image_width / 2)

						requested_colour = frame[ymid, xmid, [2,1,0]]
						print(requested_colour)

						min_colours = {}

						for key, name in webcolors.css3_hex_to_names.items():
							r_c, g_c, b_c = webcolors.hex_to_rgb(key)
							rd = (r_c - requested_colour[0]) ** 2
							gd = (g_c - requested_colour[1]) ** 2
							bd = (b_c - requested_colour[2]) ** 2
							min_colours[(rd + gd + bd)] = name

						colour_name = min_colours[min(min_colours.keys())]

						print(colour_name)
						engine.say(colour_name)
						engine.runAndWait()

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	exit()

except KeyboardInterrupt:
	cv2.destroyAllWindows()
	vs.stop()
	exit()

