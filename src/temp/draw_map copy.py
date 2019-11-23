import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pySaliencyMap as pySaliencyMap
from PIL import Image, ImageEnhance

if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except:
        path = '../images/temp.jpeg'

    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=4.0, fy=4.0)
    imgsize = img.shape
    img_width = imgsize[1]
    img_height = imgsize[0]
    print(img_height, img_width)
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    saliency_map = sm.SMGetSM(img)
    binarized_map = sm.SMGetBinarizedSM(img)
    salient_region = sm.SMGetSalientRegion(img)

    saliency_rt = np.sqrt(saliency_map)
    k = img_width//12
    if k % 2 == 0:
        k += 1
    if k > 300:
        k = 301
    print(k, 'kernel-size')
    out = cv2.GaussianBlur(saliency_rt, (k, k),
                           1/12 * (saliency_rt.shape[0] * saliency_rt.shape[1]))

    outc = out.copy()
    out *= 255
    out_img = Image.fromarray(out.astype(np.uint8))
    enhanced = ImageEnhance.Contrast(out_img)
    out_img = enhanced.enhance(2)
    enhanced = ImageEnhance.Brightness(out_img)
    out_img = enhanced.enhance(1.4)
    out2 = np.array(out_img)

    # plt.subplot(3, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Input image')
    # cv2.imshow("input",  img)
    plt.subplot(1, 2, 1), plt.imshow(saliency_map, 'gray')
    plt.title('Saliency map')
    # plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(outc, cv2.COLOR_BGR2RGB))
    # plt.title('Low contrast')
    # cv2.imshow("output", map)
    # plt.subplot(2, 2, 2), plt.imshow(binarized_map)
    # plt.title('Binarilized saliency map')
    # # cv2.imshow("Binarized", binarized_map)
    # plt.subplot(3, 2, 4), plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    # plt.title('Salient region')
    # cv2.imshow("Segmented", segmented_map)
    # plt.subplot(1, 2, 2), plt.imshow(out_img)
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
    plt.title("Draw Map")
    plt.show()
