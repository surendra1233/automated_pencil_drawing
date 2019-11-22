import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

from src.edge import get_edge_img
from src.multi_res import get_mr_img_from_rgb_img
from src.utilities import show_img
from src.multi_res_lic import get_mrl


def get_stroke_img(edge_img, mrl):
    return edge_img + mrl


def get_stroke_img_from_rgb(img):
    mr_img = get_mr_img_from_rgb_img(img)
    _, _, edge_img = get_edge_img(mr_img)
    mrl_img = get_mrl(img)
    out = get_stroke_img(edge_img, mrl_img)
    return out


if __name__ == "__main__":
    img = cv2.imread("../images/temp.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mr_img = get_mr_img_from_rgb_img(img)
    _, _, edge_img = get_edge_img(mr_img, thresh=110, thresh2=0.5)
    mrl_img = get_mrl(img)
    out = get_stroke_img(edge_img, mrl_img)
    show_img(img, splt=221,gray=False)
    show_img(edge_img, splt=222)
    show_img(mrl_img, splt=223)
    show_img(out, splt=224)
    plt.show()