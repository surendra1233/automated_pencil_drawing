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

# Usage example
# 
# saliency_map, binarized_map, salient_region, out = get_dmap(img_gray_rgb, ker=320, factor=.8)
# 
# 
def get_dmap(img, saliency_met=0, ker=None, factor=.2):
    """
    Gets the dmap from the saliency map for the input image
    """
    imgsize = img.shape
    img_width = imgsize[1]
    img_height = imgsize[0]
    # print(img_height, img_width)
    sm, saliency_map, binarized_map, salient_region = [], [], [], []
    if saliency_met == 0:
        sm = SaliencyMap.SaliencyMap(img_width, img_height)
        saliency_map = sm.SMGetSM(img)
        binarized_map = sm.SMGetBinarizedSM(img)
        salient_region = sm.SMGetSalientRegion(img)
    if saliency_met == 1:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(img)
        # saliencyMap = sa
        (suc, binarized_map) = saliency.computeBinaryMap(saliency_map)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        salient_region = np.ones(saliency_map.shape)
        saliency_map = saliency_map.astype(np.float32)
    saliency_rt = np.sqrt(saliency_map)

    k = ker
    if ker is None:
        k = img_width//12
    if k % 2 == 0:
        k += 1
    # print(k, 'kernel-size')
    # print(saliency_rt.shape)
    # print(saliency_map.dtype)

    out = cv2.GaussianBlur(saliency_rt, (k, k),
                           factor * np.prod(saliency_rt.shape[:2]))

    # print(np.amax(saliency_map), np.amin(saliency_map))
    # exit(1)
    return saliency_map, binarized_map, salient_region, out / np.amax(out)


if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except:
        path = '../images/lena.jpg'
    print(cv2.__version__)
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300))

    saliency_map, binarized_map, salient_region, out = get_dmap(
        img, saliency_met=0)

    # plt.suptitle("Saliency M")
    show_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             gray=False, splt=221, title="Input Image")
    show_img(saliency_map, title="Saliency Map", splt=222)
    show_img(binarized_map, title="Binarized saliency map", gray=False, splt=223)
    # show_img(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB), gray=False)
    show_img(out, title="Draw Map", splt=224)
    plt.show()

    saliency_map, binarized_map, salient_region, out = get_dmap(
        img, saliency_met=1)

    plt.figure()
    plt.suptitle("Method 1")
    show_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             gray=False, splt=221, title="Input Image")
    show_img(saliency_map, title="Saliency Map", splt=222)
    show_img(binarized_map, title="Binarized saliency map", gray=False, splt=223)
    # show_img(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB), gray=False)
    show_img(out, title="Draw Map", splt=224)
    # plt.show()

    # plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Input image')
    # plt.axis("off")
    # # cv2.imshow("input",  img)
    # plt.subplot(2, 2, 2), plt.imshow(saliency_map, 'gray')
    # plt.title('Saliency map')
    # plt.axis("off")
    # # cv2.imshow("output", map)
    # plt.subplot(2, 2, 3), plt.imshow(binarized_map)
    # plt.title('Binarilized saliency map')
    # plt.axis("off")
    # # cv2.imshow("Binarized", binarized_map)
    # # plt.subplot(3, 2, 4), plt.imshow(
    # #     cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    # # plt.title('Salient region')
    # # cv2.imshow("Segmented", segmented_map)
    # # out = ((out - np.amin(out)) * (255/ (np.amax(out) - np.amin(out)))).astype(np.uint8)
    #
    # plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("Draw Map")
    # plt.show()
    #
    #
