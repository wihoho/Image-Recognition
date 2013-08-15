__author__ = 'GongLi'

from Utility import *
from buildHistograms import buildHistogram

# buld vocabulary according to training data
trainData, stackOfFeatures = readData("images/training")
voc = Vocabulary(stackOfFeatures, 300, subSampling=1)
writeDataToFile("voc.pkl", voc)

# build histograms for training and testing data
buildHistogram("training")
buildHistogram("testing")

# Read in histograms
trainData = array(loadDataFromFile("Data/trainingHistogram.pkl"))
trainLabels = loadDataFromFile("Data/traininglabels.pkl")

testData = array(loadDataFromFile("Data/testingHistogram.pkl"))
testLabel = loadDataFromFile("Data/testinglabels.pkl")

# SVM
from sklearn.svm import *
clf = SVC()
clf.fit(trainData, trainLabels)
SVMResults = clf.predict(testData)

correct = sum(1.0 * (SVMResults == testLabel))
accuracy = correct / len(testLabel)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabel))+ ")"





