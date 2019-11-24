import cv2
import numpy as np
from matplotlib import pyplot as plt

from lic import (extract_region_vector_field, generate_noise_image,
                 label_regions)
from mcgp import get_gp
from utilities import show_img

def get_np_from_rgb_mc_gp(mc_gp_imgs):
    out = []
    for ind, im in enumerate(mc_gp_imgs):
        print(ind)
        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        img_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        labels, label_counts = label_regions(
            img_lab, img_lab.shape[0] * img_lab.shape[1] // 8)
        im_noise = generate_noise_image(img_gray, labels, label_counts)
        out.append(im_noise)
    return out


def get_np_vec_from_rgb(img):
    mc_gp = [img]
    get_gp(img,mc_gp,2)
    out = []
    vecs = []
    for ind, im in enumerate(mc_gp):
        print(ind)
        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        img_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        labels, label_counts = label_regions(
            img_lab, img_lab.shape[0] * img_lab.shape[1] // 8)
        vec = extract_region_vector_field(img_gray, labels, label_counts)
        im_noise = generate_noise_image(img_gray, labels, label_counts)
        out.append(im_noise)
        vecs.append(vec)
    return out, vecs

if __name__ == "__main__":
    img = cv2.imread('../images/elephant.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mc_gp = [img]
    get_gp(img,mc_gp,2)
    out = get_np_from_rgb_mc_gp(mc_gp)

    for ind, i in enumerate(out):
        show_img(i, splt=221+ind)
    plt.show()
