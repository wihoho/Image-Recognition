__author__ = 'GongLi'

from Utility import *

def buildHistogram(path, level):

    # Read in vocabulary & data
    voc = loadDataFromFile("Data/voc.pkl")
    trainData, stackOfFeatures = readData("images/"+path)

    # Transform each feature into histogram
    featureHistogram = []
    labels = []

    index = 0
    for oneImage in trainData:

        featureHistogram.append(voc.buildHistogramForEachImageAtDifferentLevels(oneImage, level))
        # featureHistogram.append(voc.buildHistogram(oneImage.descriptors))
        labels.append(oneImage.label)

        index += 1

    print "Start to store histograms"

    writeDataToFile("Report_Data/"+path+"HistogramLevel" +str(level)+ ".pkl", featureHistogram)
    writeDataToFile("Report_Data/"+path+"labels.pkl", labels)

if __name__ == '__main__':

    buildHistogram("testing", 0)
    buildHistogram("training", 0)

    buildHistogram("testing", 1)
    buildHistogram("training", 1)

    buildHistogram("testing", 2)
    buildHistogram("training", 2)
