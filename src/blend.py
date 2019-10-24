from PIL import Image
import numpy as np

def composite_paper(img, paper, alpha=.6):
    imgx = Image.fromarray(img).convert('L')
    background = Image.fromarray(paper).convert('L')
    return Image.blend(imgx, background.resize(imgx.size), alpha)
