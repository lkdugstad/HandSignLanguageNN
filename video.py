__author__ = " Lars Kristian Dugstad"
import numpy as np
import os
import cv2
import matplotlib as plt
import american_sign_language.signLanguage as sl
import tensorflow as tf

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

new_model = tf.keras.models.load_model('checkp5.ckpt')

cap = cv2.VideoCapture(0)  # Start WebCam Live-Stream
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)  # Set the window width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)  # Set the window height
i = 0
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # bounding box which captures ASL sign to be detected by the system
    cv2.rectangle(frame, (160, 120), (500, 460), (255, 0, 0), 2)
    i += 1
    # save each frame as image with PNG format
    image = cv2.imwrite('database/{index}.png'.format(index=i), frame)
    # Turn the BGR image in grey colored image
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cut out the fragment in the box of the image
    imgSliced = grey[120:460, 160:500]
    # scale the new sliced image to 28*28 pixels
    resizedImg = cv2.resize(imgSliced, (28, 28))
    imgArrNew = resizedImg.reshape(1, 28, 28, 1) / 255
    prediction = new_model.predict(imgArrNew)
    label = np.argmax(prediction)
    soft = softmax(prediction.tolist())
    #probability = str(soft)
    probability = str(soft[0][np.argmax(soft)])[:8]


    labelOut = str(sl.checkLabel(label) + " - " + str(probability))
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (180, 110)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # output the predicted label/sign on the live-stream frame
    image1 = cv2.putText(frame, labelOut, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    cv2.imshow('frame', image1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


