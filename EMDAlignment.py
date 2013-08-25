__author__ = 'GongLi'

import Utility as util
import numpy as np
from PIL import Image


def readImage(imagePath):
    img = Image.open(imagePath)
    width, height = img.size

    util.process_image_dsift(imagePath, 'temp.sift', 16, 8, False)
    l, d = util.read_features_from_file('temp.sift')

    numberOfDescriptors = l.shape[0]
    descriptors = []
    for i in range(numberOfDescriptors):
        descriptor = util.siftDescriptor(l[i][0], l[i][1], d[i])
        descriptors.append(descriptor)

    imDescriptors = util.imageDescriptors(descriptors, ["test"], width, height)

    return imDescriptors


trainImg = "images/training/Shooting/Shooting_0040.jpg"
testImg = "images/testing/Shooting/Shooting_0060.jpg"

trainImg_descriptor = readImage(trainImg)
testImg_descriptor = readImage(testImg)

voc = util.loadDataFromFile("Data/voc.pkl")
trainHistogram = voc.buildHistogramForEachImageAtDifferentLevels(trainImg_descriptor, 1)
testHistogram = voc.buildHistogramForEachImageAtDifferentLevels(testImg_descriptor, 1)

trainHistogram = trainHistogram[300:1500].reshape(4,300)
testHistogram = testHistogram[300: 1500].reshape(4, 300)

temp = 0
for i in range(4):
    t = np.linalg.norm(trainHistogram[i] - testHistogram[i])
    print str(t)

    temp += t

print "without alignment: " +str(temp / 4.0)

#
temp = 0
temp += np.linalg.norm(trainHistogram[3] - testHistogram[0])
temp += np.linalg.norm(trainHistogram[1] - testHistogram[2])
temp += np.linalg.norm(trainHistogram[0] - testHistogram[1])
temp += np.linalg.norm(trainHistogram[2] - testHistogram[3])

print "with: "+str(temp/4)






