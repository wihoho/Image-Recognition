__author__ = 'GongLi'

import numpy as np
import Utility as util
import os
import time
import subprocess as sub



def C_EMD(feature1, feature2, excutablePath):

    H = feature1.shape[0]
    I = feature2.shape[0]

    unalignedDis = 0

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

            if i == j:
                unalignedDis += distances[i][j]
    unalignedDis /= H

    groundDistanceFile = open(excutablePath+"/groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    # os.system("EarthMover")
    sub.call(excutablePath+"/EarthMoverDistance")

    # Read in EMD distance
    file = open(excutablePath+"/result", "r").readlines()

    groundDistanceFile.close()

    while True:
        try:
            os.remove(excutablePath+"/groundDistance")
            break

        except:
            time.sleep(1)
            print "groundDistance is not deleted properly!"

    alignedDis = float(file[0])
    return unalignedDis, alignedDis


if __name__ == "__main__":
    trainData = np.array(util.loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
    trainLabels = util.loadDataFromFile("Data/traininglabels.pkl")

    testData = np.array(util.loadDataFromFile("Data/testingHistogramLevel2.pkl"))
    testLabels = util.loadDataFromFile("Data/testinglabels.pkl")


    trainNum = trainData.shape[0]
    testNum = testData.shape[0]
    trainData = trainData[::, 300:1500:]
    testData = testData[::, 300:1500:]


    print "Train size: " +str(trainData.shape)
    print "Test size: " +str(testData.shape)


    # calculate EMD distance
    allData = np.vstack((trainData, testData))
    row, column = allData.shape


    aligned_allDistance = np.zeros((row, row))
    unaligned_allDistance = np.zeros((row, row))


    for i in range(0, row, 1):
        tempOne = allData[i].reshape(4, 300)
        for j in range(i+1, row, 1):
            tempTwo = allData[j].reshape(4, 300)

            unalignDis, alignedDis = C_EMD(tempOne, tempTwo, "EMDone")

            aligned_allDistance[i][j] = alignedDis
            aligned_allDistance[j][i] = alignedDis

            unaligned_allDistance[i][j] = unalignDis
            unaligned_allDistance[j][i] = unalignDis


            print "("+str(i)+","+str(j)+"): "+str(alignedDis) +"\t"+ str(unalignDis)

    util.writeDataToFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/Aligned_Alldistance.pkl", aligned_allDistance)
    util.writeDataToFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/Unaligned_Alldistance.pkl", unaligned_allDistance)

    util.writeDataToFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/train_labels.pkl", trainLabels)
    util.writeDataToFile("/Users/GongLi/PycharmProjects/ImageRecognition/Report_Data/EMD/test_labels.pkl", testLabels)

