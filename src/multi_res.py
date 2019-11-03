"""
Multi resolution image implementation using the Gaussian Pyramid
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.mcgp import get_gp
from src.draw_map import get_dmap
from src.utilities import show_img


def get_mr_img_from_rgb_img(img):
    """
    Returns the Multi resolution image for a color image
    """
    _, _, _, d_map = get_dmap(img)
    im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mc_gauss_py = [im]
    get_gp(im, mc_gauss_py, 2)
    gp_sm = np.zeros((4, *mc_gauss_py[0].shape))
    sh = mc_gauss_py[0].shape
    for i in range(4):
        gp_sm[i] = cv2.resize(mc_gauss_py[i], (sh[1], sh[0]))
    mc_gauss_py = gp_sm
    # tem = np.zeros((4, img.shape[0], img.shape[1]))
    # for i in range(4):
    #     tem[i, :, :] = mc_gauss_py[i]
    return get_mr_img(gp_sm, d_map)


def get_mr_img(gp, draw_mp):
    """
    Return an image from a gaussian pyramid
    """
    n = len(gp)
    r = draw_mp * (n-1)
    rin = r.astype(np.int)
    a = r - rin
    rin = 3 - rin

    z = tuple(np.indices((gp[0].shape[0], gp[0].shape[1])))
    # print('a', a.shape, 'r', r.shape, 'rin', rin.shape, 'gp', gp.shape, 'z', z[0].shape, z[1].shape)
    next_m = rin+1
    next_m[next_m==n] = n-1
    # print(np.amax(rin), np.amax(r), n)
    # print('gprin', gp[(rin,) + z].shape)

    out = (1-a) * gp[(rin,) + z] + a * gp[(next_m,)+z]
    return out


if __name__ == "__main__":
    img = cv2.imread("../images/temp.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (300,300))
    s_map, _, _, d_map = get_dmap(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mc_gauss_py = [img]
    get_gp(img, mc_gauss_py, 2, sm_size=True)
    tem = np.zeros((4, img.shape[0], img.shape[1]))
    for i in range(4):
        tem[i, :, :] = mc_gauss_py[i]
    # mc_gauss_py = np.concatenate(tuple(mc_gauss_py),axis=0)

    # mc_gauss_py = np.array(mc_gauss_py)
    # print('gvv',mc_gauss_py.shape)

    [print (i.shape) for i in mc_gauss_py]

    out = get_mr_img(tem, d_map)
    print(out.shape)
    show_img(img, splt=221, title="Input")
    show_img(s_map, splt=222, gray=True,title="Saliency map")
    show_img(d_map, splt=223, title="draw_map")
    show_img(out, splt=224, title="Multi resolution")
    plt.show()
