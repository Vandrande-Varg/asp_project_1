"""
import packages
"""
import pickle
import time

import cv2
import face_recognition
import imutils
import pyttsx3 as pyttsx
import webcolors
from imutils.video import VideoStream

import RPi.GPIO as G


def text_to_speech(message):
    engine.say(message)
    engine.runAndWait()


def get_closest_colour(requested_colour):
    # initialise min colours list
    min_colours = {}

    # get euclidean distance between pixel RGB values
    # and values in the webcolor RGB space
    for key, wc_name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = wc_name

    # get name of color with the smallest distance
    colour_name = min_colours[min(min_colours.keys())]

    return colour_name


def get_reloaded_encodings():
    reloaded_data = pickle.loads(open(encoding_file, "rb").read())
    return reloaded_data


G.setmode(G.BCM)
G.setup(18, G.IN, pull_up_down=G.PUD_UP)
mode = 1
encoding_file = "encodings.pickle"
cascade_file = "haarcascade_frontalface_default.xml"
engine = pyttsx.init()
# set tts wpm
engine.setProperty("rate", 180)

# load encodings
print("loading encodings")
data = pickle.loads(open(encoding_file, "rb").read())
detector = cv2.CascadeClassifier(cascade_file)
text_to_speech("loaded encodings")

# initialize video stream
print("starting video stream")
video_stream = VideoStream(usePiCamera=True).start()
# video_stream = VideoStream(src=0).start()
text_to_speech("started video stream")

try:
    while True:

        # poll for button press with de-bouncing
        if G.input(18) == False:

            time.sleep(0.08)

            if G.input(18) == False:
                start_time = time.time()

                # keep looping till button is released
                while G.input(18) == False:
                    continue

                # get time button was held down
                button_time = time.time() - start_time

                # switch modes if button held time is more than 1 sec
                if 1 < button_time < 3:

                    if mode == 1:
                        mode = 2
                        print("Colour Recognition Mode")
                        text_to_speech("Colour Recognition Mode")

                    elif mode == 2:
                        mode = 1
                        print("Face Recognition Mode")
                        text_to_speech("Face Recognition Mode")

                # reread encodings file if button held time is more than 3 sec
                elif button_time > 3:
                    data = get_reloaded_encodings()
                    print("Data Reloaded")

                # else run the current mode
                else:

                    # get frame from video stream
                    captured_frame = video_stream.read()

                    # resize image for processing
                    captured_frame = imutils.resize(captured_frame, width=500)

                    if mode == 1:

                        print("Picture Taken")

                        # BGR to gray for face detection
                        gray_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

                        # BGR to RBG for face recognition
                        rgb_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

                        # detect faces in the grayscale frame
                        rects = detector.detectMultiScale(
                            gray_frame,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30),
                            flags=cv2.CASCADE_SCALE_IMAGE,
                        )

                        # reorder and get second rectangle vertex
                        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

                        # get encodings from detected faces
                        encodings = face_recognition.face_encodings(rgb_frame, boxes)

                        # initialise list for names of recognised faces
                        names = []

                        # loop over encodings of detected faces
                        for encoding in encodings:

                            # attempt to match encoding of detected face to known encodings
                            matched_encodings = face_recognition.compare_faces(
                                data["encodings"], encoding, tolerance=0.4
                            )
                            name = "Unknown"

                            # check for matches
                            if True in matched_encodings:

                                # get indexes of matches
                                matched_indexes = [
                                    i for (i, b) in enumerate(matched_encodings) if b
                                ]

                                counts = {}

                                # loop over the matches and group postives by names
                                for i in matched_indexes:
                                    name = data["names"][i]
                                    counts[name] = counts.get(name, 0) + 1

                                # get name with largest amount of votes
                                name = max(counts, key=counts.get)

                            # say out name
                            print(name)
                            names.append(name)
                            text_to_speech(name.replace("_", " "))

                    elif mode == 2:

                        image_height = captured_frame.shape[0]  # get image height
                        image_width = captured_frame.shape[1]  # get image width

                        # get coords of middle pixel
                        y_mid = int(image_height / 2)
                        x_mid = int(image_width / 2)

                        # get RGB vlaues of pixel
                        requested_colour = captured_frame[y_mid, x_mid, [2, 1, 0]]
                        print(requested_colour)

                        closest_colour_name = get_closest_colour(requested_colour)

                        # say color name
                        print(closest_colour_name)
                        text_to_speech(closest_colour_name)

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    video_stream.stop()
    exit()
