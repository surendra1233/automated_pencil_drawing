"""
Edge image
"""

from time import sleep

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utilities import show_img
from src.multi_res import get_mr_img_from_rgb_img


def avg_strength(img, thresh=3, rev=False):
    """
    Calculates the avg_strength
    """
    kernel = 1 / 25 * np.ones((5, 5))
    avg_img = cv2.filter2D(img, -1, kernel)
    res = np.zeros(img.shape)
    res[img > (avg_img - thresh)] = 255
    return res


def get_edge_img(mr_img, thresh=190, thresh2=3):
    """
    The edge image is computed using a mean filter
    and then applying a threshold
    """
    # print(np.amax(mr_img))
    out_1 = avg_strength(mr_img, thresh=thresh2)

    kernel = 1 / 25 * np.ones((5, 5))
    avg_img = cv2.filter2D(out_1, -1, kernel)

    # out_2 = avg_strength(out_1, thresh=-20)
    res = out_1.copy()
    print(np.amax(avg_img), np.amin(avg_img))
    res[avg_img > thresh] = 255
    # diff = out_1 - avg_img
    # print(np.amax(diff), np.amin(diff))
    # print(thresh)
    # res[diff > -thresh] = 255
    # res[np.where(diff > -20)] = 255
    return out_1, avg_img, res


if __name__ == "__main__":
    img = cv2.imread("../images/temp.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (648, 292))

    out = get_mr_img_from_rgb_img(img)

    for i in range(50, 250, 10):
        plt.figure()
        plt.suptitle("E = "+str(i))
        edge_img1, edge_img2, diff = get_edge_img(out, thresh=i, thresh2=0.5)
        show_img(out, splt=221, title="Multi Resolution Image")
        show_img(edge_img1, splt=222, title="Edge Image 1")
        show_img(edge_img2, splt=223, title="Edge Image 2")
        show_img(diff, splt=224, title="Edge Image 2")
        plt.show()

    sleep(5)
