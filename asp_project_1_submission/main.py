import RPi.GPIO as G
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pyttsx3 as pyttsx
import webcolors


G.setmode(G.BCM)
G.setup(17, G.IN)
mode = 1
encoding_file = "encodings.pickle"
cascade_file = "haarcascade_frontalface_default.xml"
engine = pyttsx.init()
# set tts wpm
engine.setProperty('rate',180)

# load encodings
print("loading encodings")
data = pickle.loads(open(encoding_file, "rb").read())
detector = cv2.CascadeClassifier(cascade_file)
engine.say("loading encodings")
engine.runAndWait()

# initialize video stream
print("starting video stream")
vs = VideoStream(usePiCamera=True).start()
#vs = VideoStream(src=0).start()
engine.say("starting video stream")
engine.runAndWait()

try:
	while True:

		# get frame from video stream
		frame = vs.read()

		# poll for button press with debouncing
		if G.input(17):

			time.sleep(0.08)

			if G.input(17):
				start_time = time.time()
				
				# keep looping till button is released
				while G.input(17):
					continue
				
				# get time button was held down
				buttonTime = time.time() - start_time
				
				# switch modes if button held time is more than 1 sec 
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
				
				# else run the current mode
				else:
					# resize image for processing
					frame = imutils.resize(frame, width=500)

					if mode == 1:

						print("Picture Taken")
						
						# BGR to gray for face detection
						gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						
						# BGR to RBG for face recognition
						rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

						# detect faces in the grayscale frame
						rects = detector.detectMultiScale\
							(gray, scaleFactor=1.1, minNeighbors=5,
							 minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

						# reorder and get second rectangle vertex
						boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

						# get encodings from detected faces
						encodings = face_recognition.face_encodings(rgb, boxes)

						# loop over encodings of detected faces
						for encoding in encodings:

							# attempt to match encoding of detected face to known encodings
							matches = face_recognition.compare_faces \
								(data["encodings"], encoding, tolerance=0.4)
							name = "Unknown"

							# check for matches
							if True in matches:

								# get indexes of matches
								matchedIdxs = [i for (i, b) in enumerate(matches) if b]

								counts = {}

								# loop over the matches and group postives by names
								for i in matchedIdxs:
									name = data["names"][i]
									counts[name] = counts.get(name, 0) + 1

								# get name with largest amount of votes
								name = max(counts, key=counts.get)

							# say out name
							print(name)
							engine.say(name.replace("_", " "))
							engine.runAndWait()

					elif mode == 2:

						image_height = frame.shape[0]  # get image height
						image_width = frame.shape[1]  # get image width

						# get coords of middle pixel
						ymid = int(image_height / 2)
						xmid = int(image_width / 2)
						
						# get RGB vlaues of pixel
						requested_colour = frame[ymid, xmid, [2,1,0]]
						print(requested_colour)

						min_colours = {}

						# get euclidean distance between pixel RGB values
						# and values in the webcolor's RGB space
						for key, name in webcolors.css3_hex_to_names.items():
							r_c, g_c, b_c = webcolors.hex_to_rgb(key)
							rd = (r_c - requested_colour[0]) ** 2
							gd = (g_c - requested_colour[1]) ** 2
							bd = (b_c - requested_colour[2]) ** 2
							min_colours[(rd + gd + bd)] = name
						
						# get name of color with the smallest distance
						colour_name = min_colours[min(min_colours.keys())]
						
						# say color name
						print(colour_name)
						engine.say(colour_name)
						engine.runAndWait()

except KeyboardInterrupt:
	cv2.destroyAllWindows()
	vs.stop()
	exit()

