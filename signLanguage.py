__author__ = "Lars Kristian Dugstad"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import american_sign_language.imagesNN as imagesNN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def bildToLabel(imgArr, checkpointLoad):
    new_model = tf.keras.models.load_model(checkpointLoad)
    imgArrNew = imgArr.reshape(1, 28, 28, 1) / 255
    prediction = new_model.predict(imgArrNew)
    soft = softmax(prediction.tolist())
    print("softmax", soft)
    print("prediction", prediction)
    label = np.argmax(prediction)
    getStat(prediction)
    # getStat(soft)
    return checkLabel(label), prediction


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def getStat(prediction):
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y']
    newpred = prediction.reshape(25)
    value = np.asarray(newpred)
    print(value)
    plt.subplot(131)
    plt.bar(names, value)
    plt.show()


def checkLabel(label):
    switcher = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z",
    }
    return switcher.get(label, "invalidlabel")


def showSingleImage(arr):
    arr.reshape(28, 28)
    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.colorbar()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def showImage(train_dataRaw, x):
    '''
    plots an image from the given data at a given position
    !!!cannot call this function with the label dataset.
    :param train_dataRaw: is the raw data in shape (27455, 784) or (7172 ,784)
    :param x: which image of the data you want to look at
    :return: plots the image in plt
    '''
    arr = train_dataRaw[x].reshape(28, 28)
    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.colorbar()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def getModel():
    # Model used for the final net.
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(25)
    ])

    return model


def getDataGen():
    '''
    Data Augmentation --> basically means, we are expanding our dataset artifically
    to avoid overfitting. this means we are zooming in by 10 percent, roatate by 10°, shift horizontally
    and vertical.
    :return:
    '''
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,                              # set input mean to 0 over the dataset
        samplewise_center=False,                               # set each sample mean to 0
        featurewise_std_normalization=False,                   # divide inputs by std of the dataset
        samplewise_std_normalization=False,                    # divide each input by its std
        zca_whitening=False,                                   # apply ZCA whitening
        rotation_range=10,                                     # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,                                        # Randomly zoom image
        width_shift_range=0.1,                                 # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,                                # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,                                 # randomly flip images
        vertical_flip=False)                                   # randomly flip images
    return datagen


def runASLMnist(epochen, loadModel, saveModel, checkpointLoad, checkpointSave):
    '''
    starts the training
    1. accquires right data with getLabelDataFromCSV
    2. reshapes Data into imagesize of (28x28)
    3. build model with its layers
    4. training requirements are set
    5. training starts
    6. evaluation of test_loss and accurracy
    '''

    # Acquisition of data from CSV files
    train_label, train_dataRaw = getLabelDataFromCSV('sign_mnist_train.csv')
    print("train label and data successfully acquired")
    test_label, test_dataRaw = getLabelDataFromCSV('sign_mnist_test.csv')
    print("test label and data successfully acquired")

    # showImage(test_dataRaw, 4)

    # We perform a grayscale normalization (division by 255) to reduce the effect of illumination's differences.
    # Moreover the CNN converges faster on [0..1] data than on [0..255].
    train_dataHilf = (train_dataRaw.reshape(27455, 28, 28)) / 255
    test_data = (test_dataRaw.reshape(7172, 28, 28)) / 255

    # Reshaping the data from 1-D to 3-D as required through input by CNN's
    train_dataNew = train_dataHilf.reshape(train_dataHilf.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    # taking 10% of the train_data as a validation set
    val_data = train_dataNew[-2700:]
    val_label = train_label[-2700:]
    train_data = train_dataNew[:-2700]
    train_label = train_label[:-2700]
    # print_data = train_dataNew[-20:]
    # print_label = train_label[-20:]

    # Data Augmentation. More Information in getDataGen()
    # todo: print data before and after
    datagen = getDataGen()
    datagen.fit(train_data)
    # datagen.fit(print_data)
    # print_dataNew = datagen.flow(print_data, print_label, batch_size=128)
    # print(print_dataNew)

    # we load a model at the beginning despite the fact, that we might overthrow it while loading an older model.
    model = getModel()

    # loadingModel
    if loadModel:
        model = tf.keras.models.load_model(checkpointLoad)
        print("model geladen")
    else:
        print("no Model loaded")

    model.summary()

    # compileModel and set the learning rate.
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    model.compile(optimizer='adam', learningrate=lr[3],
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Training the model and might safe the state. We validate each model with own generated val_data to not touch
    # the test data. --> thereby we can detect overfitting as the test data has never seen before. Diminishing biases.
    if saveModel:
        history = model.fit(datagen.flow(train_data, train_label, batch_size=128),
                            epochs=epochen, validation_data=(val_data, val_label))

        model.save(checkpointSave)
        print("model.saved")
    else:
        # train Model without saving
        model.fit(datagen.flow(train_data, train_label, batch_size=128),
                  epochs=epochen, validation_data=(val_data, val_label))

    # evaluating model
    test_loss, test_acc = model.evaluate(test_data, test_label, verbose=2)
    print('\nSaveModel:', saveModel, ' Checkpoint:', checkpointSave, ' LoadedModel:', loadModel,
          ' Checkpoint:', checkpointLoad)
    print('\nTest accuracy:', test_acc, ' Test Loss:', test_loss, ' Epochen:', epochen, ' learning Rate: 0,0001')


def getLabelDataFromCSV(file):
    '''
    becomes the path to the file (maybe on windows need the actual path) and returns a Numpy Array
    :param file: file path
    :return: label and data with cut of nan as an npArray
    '''
    my_data = np.genfromtxt(file, delimiter=',')
    data = my_data[1:len(my_data), 1:len(my_data[0])]
    label = my_data[1:len(my_data), 0:1]
    return label, data


def main():
    print("did you change checkpoints?")
    epochen = 10
    loadModel = False
    saveModel = True
    checkpoints = ["checkp0.ckpt", "checkp1.ckpt",
                   "checkp2.ckpt", "checkp3.ckpt",
                   "checkp4.ckpt", "checkp5.ckpt",
                   "checkp6.ckpt"]
    checkpointLoad = checkpoints[5]
    checkpointSave = "checkpSklearn.ckpt"
    # runASLMnist(epochen, loadModel, saveModel, checkpointLoad, checkpointSave)
    imgArr = imagesNN.getOneTestImage()
    showSingleImage(imgArr)
    #label, prediction = bildToLabel(imgArr, 'checkp5.ckpt')
    # print(label)
    # getStat(prediction)


if __name__ == '__main__':
    main()
