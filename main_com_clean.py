# import the necessary packages
import RPi.GPIO as G
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pyttsx3 as pyttsx
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import webcolors


G.setmode(G.BCM)
G.setup(17, G.IN)
mode = 1
encoding_file = "encodings_2.pickle"
cascade_file = "haarcascade_frontalface_default.xml"
text_detector = "frozen_east_text_detection.pb"
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
confidence = 0.5
padding = 0.05
engine = pyttsx.init()

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encoding_file, "rb").read())
detector = cv2.CascadeClassifier(cascade_file)
engine.say("loading encodings")
engine.runAndWait()

print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(text_detector)
engine.say("loading text detector")
engine.runAndWait()

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=1).start()
# camera = cv2.VideoCapture(1)
engine.say("starting video stream")
engine.runAndWait()



def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < confidence:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return rects, confidences



try:
	# loop over frames from the video file stream
	while True:

		# grab the frame from the threaded video stream
		frame = vs.read()
		cv2.imshow("Frame", frame)

		if G.input(17):

			time.sleep(0.08)

			if G.input(17):
				start_time = time.time()

				while G.input(17):
					continue

				buttonTime = time.time() - start_time
				if buttonTime > 1:

					if mode == 1:
						mode = 2
						engine.say("Colour Recognition Mode")
						engine.runAndWait()

					elif mode == 2:
						mode = 1
						engine.say("Face Recognition Mode")
						engine.runAndWait()

				else:

					if mode == 1:
						# elif key == "t":

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

					# elif key == "c":
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

					elif mode == 3:

						image = frame
						orig = image.copy()
						(origH, origW) = image.shape[:2]

						# set the new width and height and then determine the ratio in change
						# for both the width and height
						(newW, newH) = (320, 320)
						rW = origW / float(newW)
						rH = origH / float(newH)

						# resize the image and grab the new image dimensions
						image = cv2.resize(image, (newW, newH))
						(H, W) = image.shape[:2]

						# construct a blob from the image and then perform a forward pass of
						# the model to obtain the two output layer sets
						blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
													 (123.68, 116.78, 103.94), swapRB=True, crop=False)
						net.setInput(blob)
						(scores, geometry) = net.forward(layerNames)

						# decode the predictions, then  apply non-maxima suppression to
						# suppress weak, overlapping bounding boxes
						(rects, confidences) = decode_predictions(scores, geometry)
						boxes = non_max_suppression(np.array(rects), probs=confidences)

						# initialize the list of results
						results = []

						# loop over the bounding boxes
						for (startX, startY, endX, endY) in boxes:

							# scale the bounding box coordinates based on ratios
							startX = int(startX * rW)
							startY = int(startY * rH)
							endX = int(endX * rW)
							endY = int(endY * rH)

							# apply padding
							dX = int((endX - startX) * padding)
							dY = int((endY - startY) * padding)

							# apply padding to each side of the bounding box, respectively
							startX = max(0, startX - dX)
							startY = max(0, startY - dY)
							endX = min(origW, endX + (dX * 2))
							endY = min(origH, endY + (dY * 2))

							# extract the actual padded ROI
							roi = orig[startY:endY, startX:endX]

							# tesseract config
							config = ("-l eng --oem 1 --psm 7")
							text = pytesseract.image_to_string(roi, config=config)

							# add the bounding box coordinates and OCR'd text to the list
							# of results
							results.append(((startX, startY, endX, endY), text))

						# sort the results bounding box coordinates from top to bottom
						results = sorted(results, key=lambda r:r[0][1])

						for ((startX, startY, endX, endY), text) in results:
							print(text)
							engine.say(text)
							engine.runAndWait()

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	exit()

except KeyboardInterrupt:
	cv2.destroyAllWindows()
	vs.stop()
	exit()

