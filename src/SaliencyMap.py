import numpy as np
import cv2


class SaliencyMap:
    # initialization
    def __init__(self, width, height):
        self.width = width
        self.height = height

    # extracting color channels
    def SMExtractRGBI(self, inputImage):
        # convert scale of array elements
        src = np.float32(inputImage) * 1./255
        # split
        (B, G, R) = cv2.split(src)
        # extract an intensity image
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # return
        return R, G, B, I

    # constructing a Gaussian pyramid
    def FMCreateGaussianPyr(self, src):
        dst = list()
        dst.append(src)
        for i in range(1, 9):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst

    # taking center-surround differences
    def FMCenterSurroundDiff(self, GaussianMaps):
        dst = list()
        for s in range(2, 5):
            now_size = GaussianMaps[s].shape
            now_size = (now_size[1], now_size[0])  # (width, height)
            tmp = cv2.resize(
                GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
            tmp = cv2.resize(
                GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
        return dst

    # constructing a Gaussian pyramid + taking center-surround differences
    def FMGaussianPyrCSD(self, src):
        GaussianMaps = self.FMCreateGaussianPyr(src)
        dist = self.FMCenterSurroundDiff(GaussianMaps)
        return dist

    # intensity feature maps
    def IFMGetFM(self, I):
        return self.FMGaussianPyrCSD(I)

    # Color feature maps
    def CFMGetFM(self, R, G, B):
        tmp1 = cv2.max(R, G)
        RGBMax = cv2.max(B, tmp1)
        RGBMax[RGBMax <= 0] = 0.0001
        # min(R,G)
        RGMin = cv2.min(R, G)
        # RG = (R-G)/max(R,G,B)
        RG = (R - G) / RGBMax
        # BY = (B-min(R,G)/max(R,G,B)
        BY = (B - RGMin) / RGBMax
        RG[RG < 0] = 0
        BY[BY < 0] = 0
        RGFM = self.FMGaussianPyrCSD(RG)
        BYFM = self.FMGaussianPyrCSD(BY)
        # return
        return RGFM, BYFM

    def SMGetBinarizedSM(self, src):
        self.SM = self.SMGetSM(src)
        SM_I8U = np.uint8(255 * self.SM)
        thresh, binarized_SM = cv2.threshold(
            SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binarized_SM

    def SMGetSalientRegion(self, src):
        binarized_SM = self.SMGetBinarizedSM(src)
        img = src.copy()
        mask = np.where((binarized_SM != 0), cv2.GC_PR_FGD,
                        cv2.GC_PR_BGD).astype('uint8')
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        rect = (0, 0, 1, 1)  # dummy
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel,
                    fgdModel=fgdmodel, iterCount=iterCount, mode=cv2.GC_INIT_WITH_MASK)
        mask_out = np.where((mask == cv2.GC_FGD) +
                            (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img, img, mask=mask_out)
        return output
