__author__ = 'GongLi'

from Utility import *

# build vocabulary according to training data

# print "Build Vocabulary using kmeans algorithm"
# trainData, stackOfFeatures = readData("images/training")
# voc = Vocabulary(stackOfFeatures, 300, subSampling=1)
# writeDataToFile("Data/voc.pkl", voc)

# build histograms for training and testing data

# print "Build histograms for each image at different levels"
# buildHistogram("training")
# buildHistogram("testing")

# Read in histograms

print "Histogram Intersection Kernel"
print "Histograms at level 0, 300 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel0.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel0.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")

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

print "Histograms at level 1, 1500 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel1.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel1.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")

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

print "Histograms at level 2, 6300 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel2.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")

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


print "Histograms at level 0, 300 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel0.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel0.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")


print "-----------Linear------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'linear')

print "-----------Poly------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'poly')

print "-----------RBF------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'rbf')

print ""

print "Histograms at level 1, 1500 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel1.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel1.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")


print "-----------Linear------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'linear')

print "-----------Poly------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'poly')

print "-----------RBF------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'rbf')

print ""


print "Histograms at level 2, 6300 dimensions"
trainData = array(loadDataFromFile("Data/trainingHistogramLevel2.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogramLevel2.pkl"))
testLabels = loadDataFromFile("Data/testinglabels.pkl")


print "-----------Linear------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'linear')

print "-----------Poly------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'poly')

print "-----------RBF------------------"
SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'rbf')





