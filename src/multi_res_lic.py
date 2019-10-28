import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

from src.draw_map import get_dmap
from src.lic_pyramid import get_lp
from src.multi_res import get_mr_img
from src.utilities import show_img


def get_mrl(img):
    lic_py = get_lp(img)
    _,_,_,d_map = get_dmap(img)
    lic_sm = np.zeros((4, *lic_py[0].shape))
    sh = lic_py[0].shape
    for i in range(4):
        lic_sm[i] = cv2.resize(lic_py[i], (sh[1], sh[0]))

    mr_lic_img = get_mr_img(lic_sm, d_map)
    return mr_lic_img
    # [print(i.shape) for i in lic_sm]
    # n = 4
    # r = d_map * (n-1)
    # rin = r.astype(int)
    # a = r - rin
    pass


if __name__ == "__main__":
    img = cv2.imread("../images/squirrel.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (300,300))
    out = get_mrl(img)
    show_img(out)
    plt.show()
