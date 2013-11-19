__author__ = 'GongLi'

from recognition import utils
from recognition import classification

def buildHistogram(path, level):

    # Read in vocabulary & data
    voc = utils.loadDataFromFile("Data/voc.pkl")
    trainData = utils.readImages("images/"+path)

    # Transform each feature into histogram
    featureHistogram = []
    labels = []

    index = 0
    for oneImage in trainData:

        featureHistogram.append(voc.buildHistogramForEachImageAtDifferentLevels(oneImage, level))
        labels.append(oneImage.label)

        index += 1

    utils.writeDataToFile("Data/"+path+"HistogramLevel" +str(level)+ ".pkl", featureHistogram)
    utils.writeDataToFile("Data/"+path+"labels.pkl", labels)



def main():

    # 1) build histograms

    # buildHistogram("testing", 0)
    # buildHistogram("training", 0)
    #
    # buildHistogram("testing", 1)
    # buildHistogram("training", 1)

    buildHistogram("testing", 2)
    buildHistogram("training", 2)

    # 2) classify
    print " "
    print "Level 2, 21 * (vocabulary size) dimensions, in this case, 6300 dimensions for each histogram"
    classification.SVM_Classify("Data/trainingHistogramLevel2.pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel2.pkl", "Data/testinglabels.pkl", "linear")
    classification.SVM_Classify("Data/trainingHistogramLevel2.pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel2.pkl", "Data/testinglabels.pkl", "poly")
    classification.SVM_Classify("Data/trainingHistogramLevel2.pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel2.pkl", "Data/testinglabels.pkl", "rbf")
    classification.SVM_Classify("Data/trainingHistogramLevel2.pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel2.pkl", "Data/testinglabels.pkl", "sigmoid")
    classification.SVM_Classify("Data/trainingHistogramLevel2.pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel2.pkl", "Data/testinglabels.pkl", "HI")

    classification.KNN_Classify("Data/trainingHistogramLevel2.pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel2.pkl", "Data/testinglabels.pkl")


if __name__ == "__main__":

    main()




