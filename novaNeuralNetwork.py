import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from mpl_toolkits import mplot3d
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from collections import OrderedDict
from sklearn.model_selection import train_test_split


# import os
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
# from PIL import Image


boxdim = 250
novaeList = dict()

# Match the index to the value in our classification list
# 0 - n
# 1 - y
# 2 - m
classValue = ["n", "y", "m"]


def createFitsImage(data, fileName):
    hdu = fits.PrimaryHDU(data=data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(fileName)

# Need to normalize our images to for the algorithm
def normalizeImage(img):
    # this is where we will normalize our image to values between 0 and 1
   return img/np.linalg.norm(img, ord=2, axis=1, keepdims=True)


# adds the
def createDatasetList(extracted, event, coordinates):
    extracted.update({coordinates: event})


def readInClassification(fileName):
    classified = {}
    with open(fileName) as f:
        for line in f:
            (key, val) = line.split('\t\t')
            if val == str("y\n"):
                classified[key] = 1
            elif val == str("n\n"):
                classified[key] = 0
            else: # maybe case
                classified[key] = 2
        f.close()
    return dict(OrderedDict(sorted(classified.items(), key=lambda t: t[0])))


def createList(classifyDict, eventDict):
    classify = list()
    events = list()
    for key in classifyDict:
        classify.append(classifyDict[key])
        events.append(eventDict[key])
    events, classify = reshapeList(events, classify)
    return classify, events


def reshapeList(event, classify):
    classify = np.reshape(classify, (len(classify),))
    event = np.reshape(event, (len(classify), 250, 250, 1))
    return event, classify


def plotImg(name, x_offset, y_offset, novae):
    coordinate = (str(y_offset) + "-" + str(5999 - x_offset))
    plt.style.use(astropy_mpl_style)
    imgData = get_pkg_data_filename(name)
    img = fits.getdata(imgData)
    event = np.ones((boxdim, boxdim), dtype=float)
    for i in range(0, boxdim):
        for f in range(0, boxdim):
            event[i][f] = img[5999 - (i+x_offset)][f + y_offset]
    # createFitsImage(event, str(y_offset) + "-" + str(5999 - x_offset) + ".fits")
    # event = normalizeImage(event)
    createDatasetList(novae, event, coordinate)


def createModel():
    model = Sequential()
    # input layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 1)))
    model.add(MaxPooling2D((2, 2)))
    # hidden
    model.add(Flatten())
    # output layer
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Testing the code


classifyDict = readInClassification('out.txt')  # read our classified list

# extract the data
for g in range(0, int((6000/boxdim))):
    for f in range(0, int((6000/boxdim))):
        plotImg('test1.fits', g * boxdim, f * boxdim, novaeList)

# Order the list in the same manner as the classification labels
eventDict = dict(OrderedDict(sorted(novaeList.items(), key=lambda t: t[0])))
# convert the dictionary to lists to split for testing
classifyList, eventList = createList(classifyDict, eventDict)

xTrain, xTest, yTrain, yTest = train_test_split(eventList, classifyList, test_size=0.2, random_state=42)

# Create a neural network model for training
model = createModel()
history = model.fit(xTrain, yTrain, epochs=10)
loss, accuracy = model.evaluate(xTest, yTest)

print(accuracy)







