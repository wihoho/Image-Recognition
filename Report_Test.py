# __author__ = 'GongLi'
#
# import Utility as util
#
# print "Level 0, 300 dimensions"
# util.SVM_Classify("Data/trainingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "linear")
# util.SVM_Classify("Data/trainingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "poly")
# util.SVM_Classify("Data/trainingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "rbf")
# util.SVM_Classify("Data/trainingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "HI")
# util.KNN_Classify("Data/trainingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel0.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl")
#
# print " "
# print "Level 1, 1500 dimensions"
# util.SVM_Classify("Data/trainingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "linear")
# util.SVM_Classify("Data/trainingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "poly")
# util.SVM_Classify("Data/trainingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "rbf")
# util.SVM_Classify("Data/trainingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "HI")
# util.KNN_Classify("Data/trainingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel1.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl")
#
#
# print " "
# print "Level 2, 6300 dimensions"
# util.SVM_Classify("Data/trainingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "linear")
# util.SVM_Classify("Data/trainingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "poly")
# util.SVM_Classify("Data/trainingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "rbf")
# util.SVM_Classify("Data/trainingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl", "HI")
# util.KNN_Classify("Data/trainingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/traininglabels.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testingHistogramLevel2.pkl", "/Users/GongLi/PycharmProjects/ImageRecognition/Data/testinglabels.pkl")


import Utility as util

print " "

print "Level 0, 300 dimensions"
util.SVM_Classify("Report_Data/trainingHistogramLevel0.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel0.pkl", "Report_Data/testinglabels.pkl", "linear")
util.SVM_Classify("Report_Data/trainingHistogramLevel0.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel0.pkl", "Report_Data/testinglabels.pkl", "poly")
util.SVM_Classify("Report_Data/trainingHistogramLevel0.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel0.pkl", "Report_Data/testinglabels.pkl", "rbf")
util.SVM_Classify("Report_Data/trainingHistogramLevel0.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel0.pkl", "Report_Data/testinglabels.pkl", "sigmoid")
util.SVM_Classify("Report_Data/trainingHistogramLevel0.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel0.pkl", "Report_Data/testinglabels.pkl", "HI")
util.KNN_Classify("Report_Data/trainingHistogramLevel0.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel0.pkl", "Report_Data/testinglabels.pkl")

print " "
print "Level 1, 1500 dimensions"
util.SVM_Classify("Report_Data/trainingHistogramLevel1.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel1.pkl", "Report_Data/testinglabels.pkl", "linear")
util.SVM_Classify("Report_Data/trainingHistogramLevel1.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel1.pkl", "Report_Data/testinglabels.pkl", "poly")
util.SVM_Classify("Report_Data/trainingHistogramLevel1.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel1.pkl", "Report_Data/testinglabels.pkl", "rbf")
util.SVM_Classify("Report_Data/trainingHistogramLevel1.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel1.pkl", "Report_Data/testinglabels.pkl", "sigmoid")
util.SVM_Classify("Report_Data/trainingHistogramLevel1.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel1.pkl", "Report_Data/testinglabels.pkl", "HI")
util.KNN_Classify("Report_Data/trainingHistogramLevel1.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel1.pkl", "Report_Data/testinglabels.pkl")



print " "
print "Level 2, 6300 dimensions"
util.SVM_Classify("Report_Data/trainingHistogramLevel2.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel2.pkl", "Report_Data/testinglabels.pkl", "linear")
util.SVM_Classify("Report_Data/trainingHistogramLevel2.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel2.pkl", "Report_Data/testinglabels.pkl", "poly")
util.SVM_Classify("Report_Data/trainingHistogramLevel2.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel2.pkl", "Report_Data/testinglabels.pkl", "rbf")
util.SVM_Classify("Report_Data/trainingHistogramLevel2.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel2.pkl", "Report_Data/testinglabels.pkl", "sigmoid")
util.SVM_Classify("Report_Data/trainingHistogramLevel2.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel2.pkl", "Report_Data/testinglabels.pkl", "HI")
util.KNN_Classify("Report_Data/trainingHistogramLevel2.pkl", "Report_Data/traininglabels.pkl", "Report_Data/testingHistogramLevel2.pkl", "Report_Data/testinglabels.pkl")