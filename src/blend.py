from PIL import Image
import numpy as np

def composite_paper(img, paper, alpha=.6):
    """
    Composite the input image with the given texture
    Give an rgb image as the input
    """
    imgx = Image.fromarray(img).convert('L')
    background = Image.fromarray(paper).convert('L')
    return Image.blend(imgx, background.resize(imgx.size), alpha)

if __name__ == "__main__":
    from src.edge import get_edge_img
    img = Image.open('../images/squirrel.jpeg', 'r')#.convert('L')
    background = Image.open('paper.jpg', 'r')#.convert('L')
    img_na = np.array(img)
    bg_na = np.array(background)
    _, _, edg = get_edge_img(img_na, 190)
    edg_na = edg.astype(np.uint8)

    d = composite_paper(edg_na, bg_na)
    d.show()
