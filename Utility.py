__author__ = 'GongLi'

from PIL import Image
from numpy import *
import numpy as np
import os
from pylab import *
import pickle
from scipy.cluster.vq import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import *
from pulp import *
import os

def process_image_dsift(imagename,resultname,size=20,steps=10,force_orientation=False,resize=None):

    im = Image.open(imagename).convert('L')
    if resize!=None:
        im = im.resize(resize)
    m,n = im.size

    if imagename[-3:] != 'pgm':
        #create a pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # create frames and save to temporary file
    scale = size/3.0
    x,y = meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(),y.flatten()
    frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
    savetxt('tmp.frame',frame.T,fmt='%03.3f')

    if force_orientation:
        cmmd = str("sift "+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame --orientations")
    else:
        cmmd = str("sift "+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame")

    os.environ['PATH'] += os.pathsep +'/Users/GongLi/Dropbox/FYP/PythonProject/vlfeat/bin/maci64'
    os.system(cmmd)

def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors

def plot_features(im,locs,circle=False):
    """ Show image with features. input: im (image as array),
        locs (row, col, scale, orientation of each feature). """

    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x,y,'b',linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        plot(locs[:,0],locs[:,1],'ob')
    axis('off')

def readData(folderPath):
    images = []

    stackOfFeatures = []
    for label in os.listdir(folderPath):
        if label == '.DS_Store':
            continue
        imagesPath = folderPath +"/"+ label

        for imagePath in os.listdir(imagesPath):
            # print "Extract features from " +imagePath

            imagePath = imagesPath +"/"+ imagePath
            img = Image.open(imagePath)
            width, height = img.size

            process_image_dsift(imagePath, 'temp.sift', 16, 8, False)
            l, d = read_features_from_file('temp.sift')

            numberOfDescriptors = l.shape[0]
            descriptors = []
            for i in range(numberOfDescriptors):
                descriptor = siftDescriptor(l[i][0], l[i][1], d[i])
                descriptors.append(descriptor)
                stackOfFeatures.append(descriptor.descriptor)

            imDescriptors = imageDescriptors(descriptors, label, width, height)
            images.append(imDescriptors)

    return images, array(stackOfFeatures)

def writeDataToFile(filePath, data):
    file = open(filePath, "w")
    pickle.dump(data, file)
    file.close()

def loadDataFromFile(filePath):
    file = open(filePath, 'r')
    data = pickle.load(file)
    return data

def normalizeColumn(inputArray):
    row = inputArray.shape[0]
    column = inputArray.shape[1]

    for i in range(column):
        tempSum = sum(inputArray[:,i])
        if tempSum == 0:
            continue

        for j in range(row):
            inputArray[j][i] /= float(tempSum)

def SVMclassify(trainData, trainLabels, testData, testLabels, kernelType = 'rbf', normalizeOrNot = False):

    if normalizeOrNot:
        normalizeColumn(trainData)
        normalizeColumn(testData)

    clf = SVC(kernel = kernelType)
    clf.fit(trainData, trainLabels)
    SVMResults = clf.predict(testData)

    correct = sum(1.0 * (SVMResults == testLabels))
    accuracy = correct / len(testLabels)
    print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]

    result = zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result

def EMDofImages(M, N):

    H = M.shape[0]
    I = N.shape[0]
    distances = zeros((H, I))

    for i in range(H):
        for j in range(I):
            distances[i][j] = linalg.norm(M[i] - N[j])

    # Set variables for EMD calculation

    os.environ['PATH'] += os.pathsep + '/usr/local/bin'
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))
        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # add objective function
    objectiveFunction = []
    constraint1 = []

    for i in range(H):
        for j in range(I):
            constraint1.append(variablesList[i][j])
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

    problem += lpSum(objectiveFunction)


    # add constraints
    problem += lpSum(constraint1) == 1.0
    for i in range(H):
        constraint2 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint2) <= 1.0 / H

    for j in range(I):
        constraint3 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint3) <= 1.0 / I

    # solve
    problem.writeLP("EMD.lp")

    # The problem is solved using PuLP's choice of Solver
    problem.solve(GLPK_CMD())

    # # The status of the solution is printed to the screen
    # print "Status:", LpStatus[problem.status]
    #
    # # Each of the variables is printed with it's resolved optimum value
    # for v in problem.variables():
    #     print v.name, "=", v.varValue

    # The optimised objective function value is printed to the screen
    flow = value(problem.objective)

    # means that M and N are identical, the distance should be 0
    return flow / 1.0

def C_EMD(feature1, feature2):
    os.environ['PATH'] += os.pathsep + '/usr/local/bin'

    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])


    groundDistanceFile = open("groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    os.system("/Users/GongLi/PycharmProjects/ImageRecognition/EarthMoverDistance")

    # Read in EMD distance
    file = open("result", "r").readlines()

    return float(file[0])


class Vocabulary:

    def __init__(self, stackOfDescriptors, k,  subSampling = 10):
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=300, n_init=10)
        kmeans.fit(stackOfDescriptors)
        self.vocabulary = kmeans.cluster_centers_

        # self.vocabulary, distortion = kmeans(stackOfDescriptors[::subSampling, :], k, 5)
        self.size = self.vocabulary.shape[0]

    def buildHistogram(self, imageDescriptors):
        histogram = zeros(self.size)

        stackFeatures = imageDescriptors[0].descriptor
        for descriptor in imageDescriptors[1:]:
            descriptor = descriptor.descriptor
            stackFeatures = vstack((stackFeatures, descriptor))

        codes, distance = vq(stackFeatures, self.vocabulary)

        for code in codes:
            histogram[code] += 1
        return histogram

    def buildHistogramForEachImageAtDifferentLevels(self, descriptorsOfImage, level):
        # descriptorsOfImage is an instance of class imageDescriptors (descriptors, label)
        # vocabulary is an instance of Vocabulary
        # level: 0 - vocabularySize     1 - 5 * vocabularySize      2 - 21 * vocabularySize

        width = descriptorsOfImage.width
        height = descriptorsOfImage.height
        widthStep = int(width / 4)
        heightStep = int(height / 4)

        descriptors = descriptorsOfImage.descriptors


        # level 2, a list with size = 16 to store histograms at different location
        histogramOfLevelTwo = zeros((16, 300))
        for descriptor in descriptors:
            x = descriptor.x
            y = descriptor.y
            boundaryIndex = int(x / widthStep)  + int(y / heightStep) *4

            feature = descriptor.descriptor
            shape = feature.shape[0]
            feature = feature.reshape(1, shape)

            codes, distance = vq(feature, self.vocabulary)
            histogramOfLevelTwo[boundaryIndex][codes[0]] += 1

        # level 1, based on histograms generated on level two
        histogramOfLevelOne = zeros((4, 300))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        # level 0
        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]


        if level == 0:
            return histogramOfLevelZero

        elif level == 1:
            tempOne = histogramOfLevelOne.flatten() * 0.5
            result = concatenate((histogramOfLevelZero,tempOne))
            return result

        elif level == 2:
            tempOne = histogramOfLevelOne.flatten() * 0.5
            tempTwo = histogramOfLevelTwo.flatten() * 0.25
            result = concatenate((histogramOfLevelZero, tempOne, tempTwo))
            return result

        else:
            return None


class siftDescriptor:

    def __init__(self, x, y, descriptor):
        self.x = x
        self.y = y
        self.descriptor = self.normalizeSIFT(descriptor)

    def normalizeSIFT(self, descriptor):
        descriptor = array(descriptor)
        norm = linalg.norm(descriptor)

        if norm > 1.0:
            descriptor /= float(norm)

        return descriptor


class imageDescriptors:

    def __init__(self, descriptors, label, width, height):
        self.descriptors = descriptors
        self.label = label
        self.width = width
        self.height = height
