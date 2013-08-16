__author__ = 'GongLi'

from Utility import *
import math

trainData = array(loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel2.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")


trainSize = len(trainLabels)
trainLevel2 = []
for i in range(0,trainSize, 5):
    M = trainData[i][300: 1500].reshape(4, 300)
    trainLevel2.append(M)
trainLabels = trainLabels[::5]

testSize = len(testLabels)
testLevel2 = []
for i in range(0, testSize, 5):
    M = testData[i][300: 1500].reshape(4, 300)
    testLevel2.append(M)
testLabels = testLabels[::5]

trainSize = len(trainLabels)
testSize = len(testLabels)

# Construct distance matrix
trainGramMatrix = zeros((trainSize, trainSize))
for i in range(trainSize):
    for j in range(i,trainSize, 1):
        if i == j:
            trainGramMatrix[i][j] = 1
            continue

        trainGramMatrix[i][j] = math.e ** (0 - EMDofImages(trainLevel2[i], trainLevel2[j]))
        trainGramMatrix[j][i] = trainGramMatrix[i][j]

testGramMatrix = zeros((testSize, trainSize))
for i in range(testSize):
    for j in range(trainSize):
        testGramMatrix[i][j] =  math.e ** (0 - EMDofImages(testLevel2[i], trainLevel2[j]))

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"




