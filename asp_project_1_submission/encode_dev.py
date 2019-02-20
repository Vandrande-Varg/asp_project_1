"""
import packages
"""
import concurrent.futures
import os
import pickle

import cv2
import face_recognition
from imutils import paths


def get_face_encodings(image_path):
    # read image and convert from BGR to RGB
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get detected face x,y locations
    boxes = face_recognition.face_locations(rgb, model="hog")

    # get facial encodings
    encodings = face_recognition.face_encodings(rgb, boxes)

    return encodings


def encode():
    print("starting encoding")
    encoding_file = "encodings.pickle"

    # get paths for image dataset
    print("quantifying faces")
    image_paths = list(paths.list_images("dataset"))

    # initialize lists for encodings and names
    known_encodings = []
    known_names = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        zip_list = zip(image_paths, executor.map(get_face_encodings, image_paths))
        for image_path, encodings in zip_list:
            path_index = image_paths.index(image_path) + 1
            print("processed image {}/{}".format(path_index, len(image_paths)))

            # get name from image folder
            name = image_path.split(os.path.sep)[-2]

            # add encodings found to list with respective names
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)

    # write the facial encodings + names to pickle file
    print("serializing encodings")
    data = {"encodings": known_encodings, "names": known_names}
    file = open(encoding_file, "wb")
    file.write(pickle.dumps(data))
    file.close()
    print("encoding finished")


if __name__ == "__main__":
    encode()
