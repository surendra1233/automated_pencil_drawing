import cv2
import numpy as np
from matplotlib import pyplot as plt
from utilities import show_img

def apply_gabor(img, theta, param=None):
    ksize = (31,31)
    sigma = 3.6
    lam = 10
    gamma = 0.5
    psi = 0
    ktype = cv2.CV_32F
    g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lam , gamma, psi, ktype)
    g_kernel /= 1.5*g_kernel.sum()
    out = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    # show_img(img, splt=131,title="Input")
    # show_img(g_kernel,splt=132,title="Filter")
    # show_img(out, splt=133,title="Filtered")
    # plt.show()
    return out,g_kernel
    
def gabor_pyramid(img, angles):
    res = []
    n = len(angles)
    accum = np.zeros_like(img)
    k=1
    plt.subplots_adjust(top=4,right=3)
    for angle in angles:
        # plt.figure()
        # plt.suptitle("theta = "+str(angle))
        fil,ker = apply_gabor(img,angle)
        np.maximum(accum, fil, accum)
        res.append(fil)
        # # show_img(img, splt=(n,3,k), title="input")
        # # show_img(ker, splt=(n,3,k+1), title="filter")
        # show_img(fil, splt=(n,1,k), title="output")
        k+=3
    plt.show()
    return res, accum


if __name__ == "__main__":
    img = cv2.imread('../images/elephant.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply_gabor(img,None)
    angles = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        angles.append(theta)
    out,fin = gabor_pyramid(img, angles)
    plt.figure()
    cv2.imwrite("../images/out.png", fin)
    show_img(fin)
    plt.show()
    plt.figure()
    for i in range(4):
        show_img(out[i], splt=221 + i)
    plt.show()