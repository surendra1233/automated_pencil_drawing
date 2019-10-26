"""
Drawmap\n
import the get_dmap function from this
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import src.SaliencyMap as SaliencyMap


from src.utilities import show_img

def get_dmap(img):
    imgsize = img.shape
    img_width = imgsize[1]
    img_height = imgsize[0]
    # print(img_height, img_width)
    sm, saliency_map, binarized_map, salient_region = [], [], [], []
    sm = SaliencyMap.SaliencyMap(img_width, img_height)
    saliency_map = sm.SMGetSM(img)
    binarized_map = sm.SMGetBinarizedSM(img)
    salient_region = sm.SMGetSalientRegion(img)
    saliency_rt = np.sqrt(saliency_map)

    out = cv2.GaussianBlur(saliency_rt, (11, 11),
                           1/12 * np.prod(saliency_rt.shape[:2]))

    return saliency_map, binarized_map, salient_region, out / np.amax(out)


if __name__ == '__main__':
    path = '../images/lena.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300))

    saliency_map, binarized_map, salient_region, out = get_dmap(
        img, saliency_met=0)

    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input image')
    plt.axis("off")
    # cv2.imshow("input",  img)
    plt.subplot(2, 2, 2), plt.imshow(saliency_map, 'gray')
    plt.title('Saliency map')
    plt.axis("off")
    # cv2.imshow("output", map)
    plt.subplot(2, 2, 3), plt.imshow(binarized_map)
    plt.title('Binarilized saliency map')
    plt.axis("off")
    cv2.imshow("Binarized", binarized_map)
    plt.subplot(3, 2, 4), plt.imshow(
        cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    # plt.title('Salient region')
    # cv2.imshow("Segmented", segmented_map)
    out = ((out - np.amin(out)) * (255/ (np.amax(out) - np.amin(out)))).astype(np.uint8)
    
    plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Draw Map")
    plt.show()
    
    
