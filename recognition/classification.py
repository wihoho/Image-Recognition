__author__ = 'GongLi'

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
import utils
import math

# histogram intersection kernel
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result

# classify using SVM
def SVM_Classify(trainDataPath, trainLabelPath, testDataPath, testLabelPath, kernelType):
    trainData = np.array(utils.loadDataFromFile(trainDataPath))
    trainLabels = utils.loadDataFromFile(trainLabelPath)

    testData = np.array(utils.loadDataFromFile(testDataPath))
    testLabels = utils.loadDataFromFile(testLabelPath)


    if kernelType == "HI":

        gramMatrix = histogramIntersection(trainData, trainData)
        clf = SVC(kernel='precomputed')
        clf.fit(gramMatrix, trainLabels)

        predictMatrix = histogramIntersection(testData, trainData)
        SVMResults = clf.predict(predictMatrix)
        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)
        print "SVM (Histogram Intersection): " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

    else:
        clf = SVC(kernel = kernelType)
        clf.fit(trainData, trainLabels)
        SVMResults = clf.predict(testData)

        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)
        print "SVM (" +kernelType+"): " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

# classify using KNN
def KNN_Classify(trainDataPath, trainLabelPath, testDataPath, testLabelPath):

    trainData = np.array(utils.loadDataFromFile(trainDataPath))
    trainLabels = utils.loadDataFromFile(trainLabelPath)

    testData = np.array(utils.loadDataFromFile(testDataPath))
    testLabels = utils.loadDataFromFile(testLabelPath)

    KNN = KNeighborsClassifier()
    KNN.fit(trainData, trainLabels)
    KNN_testLabels = KNN.predict(testData)

    correct = sum(1.0 * (KNN_testLabels == testLabels))
    accuracy = correct / len(testLabels)
    print "KNN: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"


# For EMD experiments
def multiClassSVM(distances, trainingIndice, testingIndice, semanticLabels, kernelType):

    distances = distances ** 2
    trainDistance = distances[np.ix_(trainingIndice, trainingIndice)]

    gamma = 1.0 / np.mean(trainDistance)
    kernelParam = []
    kernelParam.append(gamma)


    tempList = []
    tempList.append(kernelType)
    baseKernel = constructBaseKernels(tempList, kernelParam, distances)

    trainGramMatrix = baseKernel[0][np.ix_(trainingIndice, trainingIndice)]
    testGramMatrix = baseKernel[0][np.ix_(testingIndice, trainingIndice)]

    trainLabels = [semanticLabels[i] for i in trainingIndice]
    testLabels = [semanticLabels[i] for i in testingIndice]

    clf = SVC(kernel = "precomputed")
    clf.fit(trainGramMatrix, trainLabels)
    SVMResults = clf.predict(testGramMatrix)

    correct = sum(1.0 * (SVMResults == testLabels))
    accuracy = correct / len(testLabels)

    return accuracy


def constructBaseKernels(kernel_type, kernel_params, D2):

    baseKernels = []

    for i in range(len(kernel_type)):

        for j in range(len(kernel_params)):

            type = kernel_type[i]
            param = kernel_params[j]

            if type == "rbf":
                baseKernels.append(math.e **(- param * D2))
            elif type == "lap":
                baseKernels.append(math.e **(- (param * D2) ** (0.5)))
            elif type == "id":
                baseKernels.append(1.0 / ((param * D2) ** (0.5) + 1))
            elif type == "isd":
                baseKernels.append(1.0 / (param * D2 + 1))

    return baseKernels