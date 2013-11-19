__author__ = 'GongLi'

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from scipy.cluster.vq import *


class Vocabulary:

    # build vocabulary based on a stack of features
    def __init__(self, stackOfDescriptors, k,  subSampling = 10):
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit(stackOfDescriptors)

        self.vocabulary = kmeans.cluster_centers_
        self.size = self.vocabulary.shape[0]

    # convert imageDescriptors into one histogram
    def buildHistogram(self, imageDescriptors):
        histogram = np.zeros(self.size)

        stackFeatures = imageDescriptors[0].descriptor
        for descriptor in imageDescriptors[1:]:
            descriptor = descriptor.descriptor
            stackFeatures = np.vstack((stackFeatures, descriptor))

        codes, distance = vq(stackFeatures, self.vocabulary)

        for code in codes:
            histogram[code] += 1
        return histogram

    # build spatial pyramids of an image based on the attribute of level
    def buildHistogramForEachImageAtDifferentLevels(self, descriptorsOfImage, level):

        width = descriptorsOfImage.width
        height = descriptorsOfImage.height
        widthStep = int(width / 4)
        heightStep = int(height / 4)

        descriptors = descriptorsOfImage.descriptors


        # level 2, a list with size = 16 to store histograms at different location
        histogramOfLevelTwo = np.zeros((16, self.size))
        for descriptor in descriptors:
            x = descriptor.x
            y = descriptor.y
            boundaryIndex = int(x / widthStep)  + int(y / heightStep) *4

            feature = descriptor.descriptor
            shape = feature.shape[0]
            feature = feature.reshape(1, shape)

            codes, distance = vq(feature, self.vocabulary)
            histogramOfLevelTwo[boundaryIndex][codes[0]] += 1

        # level 1, based on histograms generated on level two
        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        # level 0
        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]


        if level == 0:
            return histogramOfLevelZero

        elif level == 1:
            tempZero = histogramOfLevelZero.flatten() * 0.5
            tempOne = histogramOfLevelOne.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne))
            return result

        elif level == 2:

            tempZero = histogramOfLevelZero.flatten() * 0.25
            tempOne = histogramOfLevelOne.flatten() * 0.25
            tempTwo = histogramOfLevelTwo.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne, tempTwo))
            return result

        else:
            return None