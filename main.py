# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pyttsx3 as pyttsx

encoding_file = "encodings_2.pickle"
cascade_file = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encoding_file, "rb").read())
detector = cv2.CascadeClassifier(cascade_file)

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=0).start()
engine = pyttsx.init()
time.sleep(2)

# loop over frames from the video file stream
while True:

	# grab the frame from the threaded video stream
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	#key = cv2.waitKey(1) & 0xFF
	key = input("t or q")
	#if key == ord("q"):
	if key == ("q"):
		break
	#elif key == ord("t"):
	elif key == ("t"):

		print("Picture Taken")

		# detect faces in the grayscale frame
		rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

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

	cv2.imshow("Frame", frame)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
