__author__ = 'GongLi'

from Utility import *

def buildHistogram(path):

    # Read in vocabulary & data
    voc = loadDataFromFile("Data/voc.pkl")
    trainData, stackOfFeatures = readData("images/"+path)

    # Transform each feature into histogram
    featureHistogram = []
    labels = []

    index = 0
    for oneImage in trainData:
        featureHistogram.append(voc.buildHistogram(oneImage.descriptors))
        labels.append(oneImage.label)

        index += 1

    print "Start to store histograms"

    writeDataToFile("Data/"+path+"Histogram.pkl", featureHistogram)
    writeDataToFile("Data/"+path+"labels.pkl", labels)

# buildHistogram("testing")
# buildHistogram("training")

