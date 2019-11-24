"""
Multi Contrast Gaussian Pyramid implementation
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utilities import show_img


def get_gp(im, result, N, sm_size=False):
    """
    Computes the Gaussian Pyramid and recursively appends it in the result list
    Send an empty list to get the result
    """
    if N < 0:
        return
    smoothed = cv2.GaussianBlur(im, (3, 3), 0)
    orig_size = im.shape
    resized = cv2.resize(smoothed, None, fx=0.5, fy=0.5)

    if sm_size:
        resized = cv2.resize(
            resized, (im.shape[1], im.shape[0]), fx=2.0, fy=2.0)

    resized = (0.9 * resized).astype(np.uint8)
    result.append(resized)
    get_gp(resized, result, N - 1, sm_size=sm_size)


if __name__ == "__main__":
    img = cv2.imread('../images/temp.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    out = []
    N = 3
    get_gp(img, out, N)
    print(len(out))
    for ind, i in enumerate(out):
        show_img(i, splt=421 + 2*ind, orig_contrast=True)
        show_img(i, splt=421 + 2*ind+1)
        print(i.shape)
    plt.show()
    for ind, i in enumerate(out):
        plt.subplot(221 + ind)
        plt.hist(i.ravel(), 256, [0, 256])
        print(i.shape)
    plt.show()
