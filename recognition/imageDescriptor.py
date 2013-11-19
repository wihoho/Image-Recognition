__author__ = 'GongLi'

import numpy as np

class siftDescriptor:

    def __init__(self, x, y, descriptor):
        self.x = x
        self.y = y
        self.descriptor = self.normalizeSIFT(descriptor)

    def normalizeSIFT(self, descriptor):
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)

        if norm > 1.0:
            descriptor /= float(norm)

        return descriptor


class imageDescriptors:

    def __init__(self, descriptors, label, width, height):
        self.descriptors = descriptors
        self.label = label
        self.width = width
        self.height = height

