import os
import sys

import cv2
import numpy as np
from PIL import Image


def dither(img):
    "An implementation of Floyd Steinberg dithering, error diffusion dithering"
    height, width = img.shape
    out = np.zeros(img.shape, np.uint8)
    for i in range(height-1):
        for j in range(width-1):
            old = img.item(i, j)
            new = 255 if old > 127 else 0
            out.itemset(i, j, new)
            error = old-new
            img.itemset(i, j+1, img.item(i, j+1)+error * 7/float(16))
            img.itemset(i+1, j+1, img.item(i+1, j+1)+error * 3/float(16))
            img.itemset(i+1, j, img.item(i+1, j)+error * 5/float(16))
            img.itemset(i+1, j-1, img.item(i+1, j-1)+error * 1/float(16))
    return out


def main(filename):
    image = Image.open(filename).convert('L')
    image = np.array(image)
    edge = cv2.Canny(image, 100, 200)
    edge = 255 - edge
    imdeth = dither(image)
    cv2.imshow('edge', imdeth)
    cv2.waitKey(0)
    # image.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if os.path.isfile(filename):
            main(filename)
        else:
            print('No such file', filename)
    else:
        print("Usage python main.py <filename>")
