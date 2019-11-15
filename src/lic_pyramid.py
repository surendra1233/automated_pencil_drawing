import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

from src.noise_pyramid import get_np_vec_from_rgb
from src.utilities import show_img
import vectorplot as vp


def get_lp(img):
    noise_py, vecs = get_np_vec_from_rgb(img)
    KW = 10
    kernel = np.arange(KW) + 1
    kernel = np.minimum(kernel, kernel[::-1])
    kernel = kernel / np.sum(kernel)
    kernel = kernel.astype(np.float32)
    lic_py = []
    for noise, vec in zip(noise_py, vecs):
        u, v = vec[..., 0], vec[..., 1]
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        noise_f = noise.astype(np.float32)
        data = vp.line_integral_convolution(v, u, noise_f, kernel)
        lic_py.append(data)
    return lic_py


if __name__ == "__main__":
    img = cv2.imread("../images/temp.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lic_py = get_lp(img)
    for i, im in enumerate(lic_py):
        show_img(im, splt=221+i)
    plt.show()