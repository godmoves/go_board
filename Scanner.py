import cv2
import matplotlib.pyplot as plt

from detect_line import *
from utils import imshow


img = cv2.imread('./image/board2.jpg')
print('Image size:', img.shape)

imshow(img)

paintedImg = img.copy()

paintedImg, intersectedPoints = getBoardIntersections(img, 255, 19, paintedImg)

imshow(paintedImg)

plt.show()
