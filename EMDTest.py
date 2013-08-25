__author__ = 'GongLi'

from Utility import *
import math


def buildGramMatrixWithKernel(distanceMatrix, kernelName):

    distanceMatrix = distanceMatrix ** 2
    meanValue = np.mean(distanceMatrix)

    if kernelName == "rbf":
        gramMatrix = math.e ** (0 - distanceMatrix / meanValue)
    elif kernelName == "lap":
        gramMatrix = math.e ** (0 - (distanceMatrix / meanValue) ** (0.5))
    elif kernelName == "id":
        gramMatrix = 1.0 / ((distanceMatrix/ meanValue) ** (0.5) + 1.0)
    elif kernelName == "isd":
        gramMatrix = 1.0 / (distanceMatrix/ meanValue + 1)

    return gramMatrix

trainData = array(loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel2.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")

step = 2

trainSize = len(trainLabels)
trainLevel2 = []
for i in range(1, trainSize, step):
    M = trainData[i][300: 1500]
    trainLevel2.append(M)
trainLabels = trainLabels[1::step]

testSize = len(testLabels)
testLevel2 = []
for i in range(0, testSize, step):
    M = testData[i][300: 1500]
    testLevel2.append(M)
testLabels = testLabels[0::step]

trainSize = len(trainLabels)
testSize = len(testLabels)

print "-----------RBF with complete data------------------"
SVMclassify(array(trainLevel2), trainLabels, array(testLevel2), testLabels, kernelType = 'rbf')

print "-----------Distances -> Gram matrix implemented by RBF------------------"
# Direct distance
trainDistance = zeros((trainSize, trainSize))
for i in range(trainSize):
    M = trainLevel2[i].reshape(4,300)

    for j in range(i, trainSize, 1):
        if i == j:
            trainDistance[i][j] = 0
            continue

        N = trainLevel2[j].reshape(4, 300)
        temp = 0
        for m in range(4):
            temp += linalg.norm(M[m] - N[m])

        trainDistance[i][j] = temp / 4.0
        trainDistance[j][i] = trainDistance[i][j]


# normalize the distances
# meanValue = np.mean(trainGramMatrix)
# trainGramMatrix = trainGramMatrix / meanValue
# trainGramMatrix = math.e ** (0 - trainGramMatrix ** 2)

testDistance = zeros((testSize, trainSize))
for i in range(testSize):
    M = testLevel2[i].reshape(4, 300)
    for j in range(trainSize):
        N = trainLevel2[j].reshape(4, 300)

        temp = 0
        for m in range(4):
            temp += linalg.norm(M[m] - N[m])

        testDistance[i][j] = temp / 4.0

# normalize the distances
# meanValue = np.mean(testGramMatrix)
# testGramMatrix = testGramMatrix / meanValue
# testGramMatrix = math.e ** (0 - testGramMatrix ** 2)


###################################### RBF
kernelName = "rbf"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName+": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

###################################### LAP

kernelName = "lap"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName+": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

###################################### ID
kernelName = "id"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName+": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

###################################### ISD
kernelName = "isd"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName+": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"
######################################


# Construct EMD distance classification
trainDistance = zeros((trainSize, trainSize))
for i in range(trainSize):
    M = trainLevel2[i].reshape(4, 300)
    for j in range(i,trainSize, 1):
        if i == j:
            trainDistance[i][j] = 0
            continue

        N = trainLevel2[j].reshape(4, 300)

        trainDistance[i][j] = C_EMD(M, N)
        trainDistance[j][i] = trainDistance[i][j]


testDistance = zeros((testSize, trainSize))
for i in range(testSize):
    M = testLevel2[i].reshape(4, 300)
    for j in range(trainSize):
        N = trainLevel2[j].reshape(4, 300)
        testDistance[i][j] =  C_EMD(M, N)

################################################rbf
kernelName = "rbf"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName +": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

################################################lap
kernelName = "lap"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName +": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

################################################id
kernelName = "id"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName +": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

################################################isd
kernelName = "isd"
trainGramMatrix = buildGramMatrixWithKernel(trainDistance, kernelName)
testGramMatrix = buildGramMatrixWithKernel(testDistance,  kernelName)

# build up the SVM model
clf = SVC(kernel='precomputed')
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print kernelName +": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"