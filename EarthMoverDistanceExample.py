__author__ = 'GongLi'

from SpatialPyramidExample import buildHistogram
from recognition import utils
from recognition import EarthMoverDistance
from recognition import classification
import numpy as np



def main():

    # 1) build histograms at level 1
    buildHistogram("testing", 2)
    buildHistogram("training", 2)

    # 2) calculate EMD distance matrix

    trainData = np.array(utils.loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
    trainLabels = utils.loadDataFromFile("Data/traininglabels.pkl")

    testData = np.array(utils.loadDataFromFile("Data/testingHistogramLevel2.pkl"))
    testLabels = utils.loadDataFromFile("Data/testinglabels.pkl")

    trainData = trainData[::, 300:1500:]
    testData = testData[::, 300:1500:]

    # calculate EMD distance
    allData = np.vstack((trainData, testData))
    row, column = allData.shape

    aligned_allDistance = np.zeros((row, row))
    unaligned_allDistance = np.zeros((row, row))


    for i in range(0, row, 1):
        tempOne = allData[i].reshape(4, 300)
        for j in range(i+1, row, 1):
            tempTwo = allData[j].reshape(4, 300)

            unalignDis, alignedDis = EarthMoverDistance.C_EMD(tempOne, tempTwo, "EMDone")

            aligned_allDistance[i][j] = alignedDis
            aligned_allDistance[j][i] = alignedDis

            unaligned_allDistance[i][j] = unalignDis
            unaligned_allDistance[j][i] = unalignDis


            print "("+str(i)+","+str(j)+"): "+str(alignedDis) +"\t"+ str(unalignDis)

    # 3) classify
    allLabels = trainLabels + testLabels
    train, test = utils.randomGenerateIndices(allLabels, 0.5)

    aligned_accuracyDict = {}
    unaligned_accuracyDict = {}
    for type in ["rbf", "lap", "id", "isd"]:
        aligned_accuracyDict[type] = []
        unaligned_accuracyDict[type] = []

    for type in ["rbf", "lap", "id", "isd"]:
        alignedAccuracy = classification.multiClassSVM(aligned_allDistance, train, test, allLabels, type)
        unalignedAccuracy = classification.multiClassSVM(unaligned_allDistance, train, test, allLabels, type)

        aligned_accuracyDict[type].append(alignedAccuracy)
        unaligned_accuracyDict[type].append(unalignedAccuracy)


    print " "
    print "Aligned ----------"
    for type in ["rbf", "lap", "id", "isd"]:
        alignedTest = aligned_accuracyDict[type]
        alignedTest = np.array(alignedTest)

        mean = np.mean(alignedTest)
        std = np.std(alignedTest)

        print type +": "+ str(mean)    +u"\u00B1"+str(std)


    print " "
    print "Unaligned --------------"
    for type in ["rbf", "lap", "id", "isd"]:
        alignedTest = unaligned_accuracyDict[type]
        alignedTest = np.array(alignedTest)

        mean = np.mean(alignedTest)
        std = np.std(alignedTest)

        print type +": "+ str(mean)    +u"\u00B1"+str(std)


if __name__ == "__main__":
    main()