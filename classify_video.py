from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import pickle
import numpy as np
import cv2
import argparse

cap = cv2.VideoCapture(0)
cap.set(28, 255) 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())


# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())


while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    output = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (72, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]

    label = "{}: {:.2f}%".format(label, proba[idx] * 100)
    output = imutils.resize(output, width=400)

    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

    cv2.imshow("Output", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()