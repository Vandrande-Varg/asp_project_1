from imutils import paths
import face_recognition
import pickle
import cv2
import os


def encode():
    print("starting encoding")
    encoding_file = "encodings.pickle"

    # get paths for image dataset
    print("quantifying faces")
    imagePaths = list(paths.list_images("dataset"))

    # initialize lists for encodings and names
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):

        # get name from image folder
        print("processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # read image and convert from BGR to RGB
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get detected face x,y locations
        boxes = face_recognition.face_locations(rgb, model="hog")

        # get facial encodings
        encodings = face_recognition.face_encodings(rgb, boxes)

        # add encodings found to list with respective names
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

            # write the facial encodings + names to pickle file
    print("serializing encodings")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(encoding_file, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("encoding finished")


if __name__ == "__main__":
    encode()

