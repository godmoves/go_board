import cv2
import matplotlib.pyplot as plt

from detect_line import getBoardIntersections
from utils import imshow
from detect_board import getWarpedImg

img = cv2.imread('./image/board8.jpg')
print('Image size:', img.shape)

imshow(img, 'original image')

warpedImg = getWarpedImg(img)

imshow(warpedImg, 'warpedImg')

paintedImg, intersectedPoints = getBoardIntersections(warpedImg, 255, 19)

imshow(paintedImg, 'paintedImg')

if len(intersectedPoints) > 4:
    pass

plt.show()
