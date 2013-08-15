__author__ = 'GongLi'

from PIL import Image
from numpy import *
import os
from pylab import *
import pickle
from scipy.cluster.vq import *
from sklearn.cluster import KMeans

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
        imagesPath = folderPath +"/"+ label

        for imagePath in os.listdir(imagesPath):
            print "Extract features from " +imagePath

            imagePath = imagesPath +"/"+ imagePath
            process_image_dsift(imagePath, 'temp.sift', 16, 8, False)
            l, d = read_features_from_file('temp.sift')

            numberOfDescriptors = l.shape[0]
            descriptors = []
            for i in range(numberOfDescriptors):
                descriptor = siftDescriptor(l[i][0], l[i][1], d[i])
                descriptors.append(descriptor)
                stackOfFeatures.append(descriptor.descriptor)

            imDescriptors = imageDescriptors(descriptors, label)
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


class Vocabulary:

    def __init__(self, stackOfDescriptors, k,  subSampling = 10):
        # kmeans = KMeans(init='k-means++', n_clusters=300, n_init=10)
        # kmeans.fit(stackOfDescriptors)
        # self.vocabulary = kmeans.cluster_centers_

        self.vocabulary, distortion = kmeans(stackOfDescriptors[::subSampling, :], k, 1)
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

    def __init__(self, descriptors, label):
        self.descriptors = descriptors
        self.label = label
