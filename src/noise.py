import cv2
import numpy as np
from matplotlib import pyplot as plt

from mcgp import get_gp
from utilities import show_img

def get_np(mc_gp, th = 10):
    n = len(mc_gp)
    out_np = np.ones((n, *mc_gp[0].shape))
    for ind, im in enumerate(mc_gp):
        val = 1 + ((ind+1) / n) *(mc_gp[ind] - 1)
        out_np[ind][np.where(val < th)] = 0
    return out_np

if __name__ == "__main__":
    img = cv2.imread("../images/fig.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mc_gp = [img]
    get_gp(img, mc_gp, 2)
    for th in range(50, 200, 10):
        plt.figure()
        out = get_np(mc_gp, th=th)
        plt.suptitle("th = "+str(th))
        print(out[0])
        [show_img(out[i], splt=221 + i, gray=False) for i in range(4)]
        plt.show()
