__author__ = 'GongLi'

from Utility import *
import math

trainData = array(loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel2.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")

step = 5

trainSize = len(trainLabels)
trainLevel2 = []
for i in range(0, trainSize, step):
    M = trainData[i][300: 1500]
    trainLevel2.append(M)
trainLabels = trainLabels[::step]

testSize = len(testLabels)
testLevel2 = []
for i in range(0, testSize, step):
    M = testData[i][300: 1500]
    testLevel2.append(M)
testLabels = testLabels[::step]

trainSize = len(trainLabels)
testSize = len(testLabels)

# print "-----------RBF------------------"
SVMclassify(array(trainLevel2), trainLabels, array(testLevel2), testLabels, kernelType = 'rbf')

# Direct distance
trainGramMatrix = zeros((trainSize, trainSize))
for i in range(trainSize):
    M = trainLevel2[i]
    for j in range(i, trainSize, 1):
        if i == j:
            trainGramMatrix[i][j] = 0
            continue

        N = trainLevel2[j]
        trainGramMatrix[i][j] = linalg.norm(M - N)

trainGramMatrix = math.e ** (0 - trainGramMatrix ** 2)

testGramMatrix = zeros((testSize, trainSize))
for i in range(testSize):
    M = testLevel2[i]
    for j in range(trainSize):
        N = trainLevel2[j]
        testGramMatrix[i][j] =  linalg.norm(M - N)

testGramMatrix = math.e ** (0 - testGramMatrix ** 2)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

# Construct distance matrix
trainGramMatrix = zeros((trainSize, trainSize))
for i in range(trainSize):
    M = trainLevel2[i].reshape(4, 300)
    for j in range(i,trainSize, 1):
        if i == j:
            trainGramMatrix[i][j] = 0
            continue

        N = trainLevel2[j].reshape(4, 300)

        trainGramMatrix[i][j] = EMDofImages(M, N)
        trainGramMatrix[j][i] = trainGramMatrix[i][j]

trainGramMatrix = math.e ** (0 - trainGramMatrix ** 2)

testGramMatrix = zeros((testSize, trainSize))
for i in range(testSize):
    M = testLevel2[i].reshape(4, 300)
    for j in range(trainSize):
        N = trainLevel2[j].reshape(4, 300)
        testGramMatrix[i][j] =  EMDofImages(M, N)

testGramMatrix = math.e ** (0 - testGramMatrix ** 2)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"
