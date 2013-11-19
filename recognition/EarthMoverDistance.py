__author__ = 'GongLi'

import subprocess as sub
import numpy as np
import time
import os

def C_EMD(feature1, feature2, excutablePath):

    H = feature1.shape[0]
    I = feature2.shape[0]

    unalignedDis = 0

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

            if i == j:
                unalignedDis += distances[i][j]
    unalignedDis /= H

    groundDistanceFile = open(excutablePath+"/groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    # os.system("EarthMover")
    sub.call(excutablePath+"/EarthMoverDistance")

    # Read in EMD distance
    file = open(excutablePath+"/result", "r").readlines()

    groundDistanceFile.close()

    while True:
        try:
            os.remove(excutablePath+"/groundDistance")
            break

        except:
            time.sleep(1)
            print "groundDistance is not deleted properly!"

    alignedDis = float(file[0])
    return unalignedDis, alignedDis
