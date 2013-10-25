__author__ = 'GongLi'

import math
import numpy as np
from sklearn.svm import SVC
import Utility as util
import random

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


if __name__ == "__main__":

    alignedDistance = util.loadDataFromFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/Aligned_Alldistance.pkl")
    unalignedDistance = util.loadDataFromFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/Unaligned_Alldistance.pkl")

    trainLabels = util.loadDataFromFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/train_labels.pkl")
    testLabels = util.loadDataFromFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/test_labels.pkl")

    trainNum = len(trainLabels)
    testNum = len(testLabels)
    trainIndices = [i for i in range(0, trainNum, 1)]
    testIndices = [i for i in range(trainNum, trainNum +testNum,1)]
    allLabels = [i for i in trainLabels]


    for i in testLabels:
        allLabels.append(i)


    aligned_accuracyDict = {}
    unaligned_accuracyDict = {}
    for type in ["rbf", "lap", "id", "isd"]:
        aligned_accuracyDict[type] = []
        unaligned_accuracyDict[type] = []



    # while True:

    train, test = randomGenerateIndices(allLabels, 0.5)

    for type in ["rbf", "lap", "id", "isd"]:
        alignedAccuracy = multiClassSVM(alignedDistance, train, test, allLabels, type)
        unalignedAccuracy = multiClassSVM(unalignedDistance, train, test, allLabels, type)

        aligned_accuracyDict[type].append(alignedAccuracy)
        unaligned_accuracyDict[type].append(unalignedAccuracy)

        # print type +"\t"+str(alignedAccuracy)+"\t"+str(unalignedAccuracy)


    alignedCheck = 0
    unalignedCheck = 0
    print " "
    print "Aligned ----------"
    for type in ["rbf", "lap", "id", "isd"]:
        alignedTest = aligned_accuracyDict[type]
        alignedTest = np.array(alignedTest)

        mean = np.mean(alignedTest)
        std = np.std(alignedTest)

        print type +": "+ str(mean)    +u"\u00B1"+str(std)

        if type == "rbf":
            alignedCheck = mean



    print " "
    print "Unaligned --------------"
    for type in ["rbf", "lap", "id", "isd"]:
        alignedTest = unaligned_accuracyDict[type]
        alignedTest = np.array(alignedTest)

        mean = np.mean(alignedTest)
        std = np.std(alignedTest)

        print type +": "+ str(mean)    +u"\u00B1"+str(std)

        if type == "rbf":
            unalignedCheck = mean

        # if alignedCheck > unalignedCheck:
        #     break



