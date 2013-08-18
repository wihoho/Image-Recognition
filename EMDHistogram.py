__author__ = 'GongLi'

from Utility import *

globalCount = 0

def EarthMoverHistogram(M, N):
    # M: 4 * 300
    # N: 4 * 300
    # Find a binary match which maximizes histogram intersection

    H = M.shape[0]
    I = N.shape[0]
    intersections = zeros((H,I))

    for i in range(H):
        for j in range(I):
            intersections[i][j] = sum(np.minimum(M[i], N[j]))

    from pulp import *
    import os
    os.environ['PATH'] += os.pathsep + '/usr/local/bin'
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))
        variablesList.append(tempList)

    problem = LpProblem("EMD of histogram intersection", LpMaximize)

    # objective function
    objectiveFunction = []
    for i in range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * intersections[i][j])
    problem += lpSum(objectiveFunction)

    # two constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) == 1.0

    for j in range(I):
        constraint2 =[variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) == 1.0

    problem.writeLP("EMD_Histogram.lp")
    problem.solve(GLPK_CMD())

    flow = value(problem.objective)


    # special match
    global globalCount
    if variablesList[0][0].varValue != 1.0 or variablesList[1][1].varValue != 1.0 or variablesList[2][2].varValue != 1.0 or variablesList[3][3].varValue != 1.0:
        globalCount += 1

    return flow / 4.0

print "Histograms at level 1, 1200 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel1.pkl"))[:, 300:]
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel1.pkl"))[:, 300:]
testLabels = loadDataFromFile("Data/testinglabels.pkl")


trainData = trainData[::2,:]
testData = testData[::2, :]

trainLabels = trainLabels[::2]
trainSize = len(trainLabels)

testLabels = testLabels[::2]
testSize = len(testLabels)

gramMatrix = histogramIntersection(trainData, trainData)
clf = SVC(kernel='precomputed')
clf.fit(gramMatrix, trainLabels)

# predict
predictMatrix = histogramIntersection(testData, trainData)
SVMResults = clf.predict(predictMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

print ""



gramMatrix = zeros((trainSize, trainSize))
for i in range(trainSize):
    M = trainData[i].reshape(4, 300)
    for j in range(i, trainSize, 1):
        N = trainData[j].reshape(4, 300)
        gramMatrix[i][j] = EarthMoverHistogram(M, N)

        if i != j:
            gramMatrix[j][i] = gramMatrix[i][j]

clf = SVC(kernel='precomputed')
clf.fit(gramMatrix, trainLabels)

# predict
predictMatrix = zeros((testSize, trainSize))
for i in range(testSize):
    M = testData[i].reshape(4, 300)
    for j in range(trainSize):
        N = trainData[j].reshape(4, 300)
        predictMatrix[i][j] = EarthMoverHistogram(M, N)

SVMResults = clf.predict(predictMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

print ""







