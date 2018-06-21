import cv2
import matplotlib.pyplot as plt
import random


def imshow(img, title='Image'):
    plt.figure(random.choice(range(10000)))
    plt.title(title)
    if img.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
        # plt,imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.show()
