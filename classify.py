__author__ = 'GongLi'
import Utility
from numpy import array

# Read in data
trainData = array(Utility.loadDataFromFile("Data/trainingHistogram.pkl"))
trainLabels = Utility.loadDataFromFile("Data/traininglabels.pkl")

testData = array(Utility.loadDataFromFile("Data/testingHistogram.pkl"))
testLabel = Utility.loadDataFromFile("Data/testinglabels.pkl")

# SVM
from sklearn.svm import *
clf = SVC()
clf.fit(trainData, trainLabels)
SVMResults = clf.predict(testData)

correct = sum(1.0 * (SVMResults == testLabel))
accuracy = correct / len(testLabel)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabel))+ ")"




