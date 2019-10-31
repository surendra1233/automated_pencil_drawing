import numpy as np
from matplotlib import pyplot as plt


def show_img(img, gray=True, splt=111, title="Image", axis="off", orig_contrast=False):
    if type(splt) == int:
        plt.subplot(splt)
        # plt.axis([0, img.shape[1], 0, img.shape[0]])
        plt.axis(axis)
        plt.title(title)
    else:

        plt.subplot(splt[0], splt[1], splt[2])
        plt.axis(axis,sharex=True)
        plt.title(title)

    if orig_contrast:
        min_c = 0
        max_c = 256

    else:
        min_c = np.amin(img)
        max_c = np.amax(img)

    if gray:
        plt.imshow(img, "gray", vmin=min_c, vmax=max_c)
    else:
        plt.imshow(img, vmin=min_c, vmax=max_c)
