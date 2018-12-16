# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pyttsx3 as pyttsx
import numpy as np
from PIL import Image
from colorlabeler import ColorLabeler
import webcolors

# lower = {'red': (166, 84, 141),
# 		 'green': (66, 122, 129),
# 		 'blue': (97, 100, 117),
# 		 'yellow': (23, 59, 119),
# 		 'orange': (0, 50, 80)}  # assign new item lower['blue'] = (93, 10, 0)
#
# upper = {'red': (186, 255, 255),
# 		 'green': (86, 255, 255),
# 		 'blue': (117, 255, 255),
# 		 'yellow': (54, 255, 255),
# 		 'orange': (20, 255, 255)}
#
# colors = {'red': (0, 0, 255),
# 		  'green': (0, 255, 0),
# 		  'blue': (255, 0, 0),
# 		  'yellow': (0, 255, 217),
# 		  'orange': (0, 140, 255)}

encoding_file = "encodings_2.pickle"
cascade_file = "haarcascade_frontalface_default.xml"
engine = pyttsx.init()

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encoding_file, "rb").read())
detector = cv2.CascadeClassifier(cascade_file)
engine.say("loading encodings")
engine.runAndWait()

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=1).start()
# camera = cv2.VideoCapture(1)
engine.say("starting video stream")
engine.runAndWait()

try:
	# loop over frames from the video file stream
	while True:

		# grab the frame from the threaded video stream
		frame = vs.read()
		# (grabbed, frame) = camera.read()
		frame = imutils.resize(frame, width=500)
		cv2.imshow("Frame", frame)

		key = cv2.waitKey(1) & 0xFF
		# key = input("t or q")
		if key == ord("q"):
			# if key == "q":
			break
		elif key == ord("t"):
			# elif key == "t":

			print("Picture Taken")

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

		# elif key == "c":
		elif key == ord("c"):

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
			#
			# size = 30
			#
			# x1 = xmid - size
			# y1 = ymid + size
			#
			# x2 = xmid + size
			# y2 = ymid - size
			#
			# clone = blurred.copy()
			# crop_img = clone[y2:y1, x1:x2]
			#
			# res = cv2.copyMakeBorder(crop_img, ymid, ymid, xmid, xmid, cv2.BORDER_CONSTANT)
			#
			# cv2.imshow("res", res)
			#
			# gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
			# lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
			# thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
			#
			# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# cnts = imutils.grab_contours(cnts)
			#
			# cl = ColorLabeler()
			#
			# for c in cnts:
			# 	color = cl.label(lab, c)
			# 	print(color)
			# 	engine.say(color)
			# 	engine.runAndWait()




			# hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
			#
			# for key, value in upper.items():
			#
			# 	# kernel = np.ones((9, 9), np.uint8)
			# 	mask = cv2.inRange(hsv, lower[key], upper[key])
			# 	# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
			# 	# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
			#
			# 	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
			#
			# 	# only proceed if at least one contour was found
			# 	if len(cnts) > 0:
			#
			# 		c = max(cnts, key=cv2.contourArea)
			# 		((x, y), radius) = cv2.minEnclosingCircle(c)
			#
			# 		if radius > 0.5:
			# 			cv2.circle(hsv, (int(x), int(y)), int(radius), colors[key], 2)
			# 			cv2.putText(hsv,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
			# 			cv2.imshow("Frame", hsv)
			# 			print(key)
			# 			engine.say(key)
			# 			engine.runAndWait()


	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	exit()

except KeyboardInterrupt:
	cv2.destroyAllWindows()
	vs.stop()
	exit()
