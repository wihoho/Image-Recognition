__author__ = 'GongLi'

import pickle
import Image
import os
from . import imageDescriptor
import numpy as np
import random

# serialize python object
def writeDataToFile(filePath, data):
    file = open(filePath, "wb")
    pickle.dump(data, file)
    file.close()

# deserialize
def loadDataFromFile(filePath):
    file = open(filePath, 'rb')
    data = pickle.load(file)
    return data

# read a folder of images and return their features
def readImages(folderPath):
    images = []

    stackOfFeatures = []
    for label in os.listdir(folderPath):
        if label == '.DS_Store':
            continue
        imagesPath = folderPath +"/"+ label

        for imagePath in os.listdir(imagesPath):
            if imagePath == ".DS_Store":
                continue

            imagePath = imagesPath +"/"+ imagePath
            print "Extract SIFT features of " + imagePath

            img = Image.open(imagePath)
            width, height = img.size

            process_image_dsift(imagePath, 'temp.sift', 16, 8, False)
            l, d = read_features_from_file('temp.sift')

            numberOfDescriptors = l.shape[0]
            descriptors = []
            for i in range(numberOfDescriptors):
                descriptor = imageDescriptor.siftDescriptor(l[i][0], l[i][1], d[i])
                descriptors.append(descriptor)
                stackOfFeatures.append(descriptor.descriptor)

            imDescriptors = imageDescriptor.imageDescriptors(descriptors, label, width, height)
            images.append(imDescriptors)

    return images

# extract SIFT features from an image
def process_image_dsift(imagename,resultname,size=20,steps=10,force_orientation=False,resize=None):

    im = Image.open(imagename).convert('L')
    if resize!=None:
        im = im.resize(resize)
    m,n = im.size

    if imagename[-3:] != 'pgm':
        #create a pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # create frames and save to temporary file
    scale = size/3.0
    x,y = np.meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(),y.flatten()
    frame = np.array([xx,yy,scale * np.ones(xx.shape[0]), np.zeros(xx.shape[0])])
    np.savetxt('tmp.frame',frame.T,fmt='%03.3f')

    if force_orientation:
        cmmd = str("sift "+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame --orientations")
    else:
        cmmd = str("sift "+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame")

    os.environ['PATH'] += os.pathsep +'/Users/GongLi/Dropbox/FYP/PythonProject/vlfeat/bin/maci64'
    os.system(cmmd)

def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    f = np.loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def randomGenerateIndices(AllLabels, percentage):

    totalNum = len(AllLabels)
    categories = set(AllLabels)
    totalClasses = len(categories)

    step = int(totalNum / totalClasses * percentage)



    classIndices = []
    for c in categories:
        tempList = []
        for i in range(len(AllLabels)):
            if AllLabels[i] == c:
                tempList.append(i)
        classIndices.append(tempList)


    trainIndi = []
    for item in classIndices:
        tempList = random.sample(item, step)
        for j in tempList:
            trainIndi.append(j)

    testIndi = [i for i in range(totalNum)]
    for i in trainIndi:
        testIndi.remove(i)


    return trainIndi, testIndi