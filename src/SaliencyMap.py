import cv2
import numpy as np


class SaliencyMap:
    # initialization
    def __init__(self, width, height):
        self.width = width
        self.height = height

    # extracting color channels
    def SMExtractRGBI(self, inputImage):
        # convert scale of array elements
        src = np.float32(inputImage) * 1./255
        # split
        (B, G, R) = cv2.split(src)
        # extract an intensity image
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # return
        return R, G, B, I

    def FMCreateGaussianPyr(self, src):
        dst = list()
        dst.append(src)
        for i in range(1,4):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst
