import cv2
from utils import imshow
import numpy as np
from numpy.random import randint


def calc_bbox_area(b):
    return b[2] * b[3]


def calc_bbox_area_sum(bboxes):
    area_sum = 0
    for b in bboxes:
        area = calc_bbox_area(b)
        # print('area', area)
        area_sum += area

    return area_sum


def delete_small_component(img, min_size=250):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    # min_size = min_size

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2


def automatic_warp(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v_channel = cv2.split(imgHSV)

    source_channel = v_channel[1]
    source_channel = cv2.medianBlur(source_channel, 3)

    # imshow(source_channel, 'source_channel')

    # cannyImg = cv2.Canny(source_channel, 0, 250)
    # cannyImg = delete_small_component(cannyImg)
    # imshow(cannyImg, 'cannyImg')

    threhold, source_channel = cv2.threshold(source_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # + cv2.THRESH_OTSU

    # source_channel = 255 - source_channel + cannyImg

    imshow(source_channel, 'source_channel threshold')

    clone = source_channel.copy()
    clone, contours, hierarchy = cv2.findContours(clone, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))

    if len(contours) == 0:
        print('No contours found!')
        return None

    factor = 0.065
    bboxes = []
    new_contours = []
    for c in contours:
        # print('contour:', c.shape)
        b = cv2.approxPolyDP(c, cv2.arcLength(c, True) * factor, True)
        new_contours.append(b)
        # print('box:', cv2.boundingRect(b))
        bboxes.append(cv2.boundingRect(b))

    bbox_area_sum = calc_bbox_area_sum(bboxes)
    mean_area = bbox_area_sum / len(contours)

    assert len(bboxes) == len(new_contours)

    # print(len(new_contours))
    # print(hierarchy.shape)

    # print(mean_area)

    edge_factor = 0.995
    max_width = img.shape[1] * edge_factor
    max_height = img.shape[0] * edge_factor

    drawing = img.copy()
    for i, _ in enumerate(new_contours):
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.drawContours(drawing, new_contours, i, color, 2, 8, hierarchy, 0, (0, 0))

    for b in bboxes:
        cv2.rectangle(drawing, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (255, 255, 255), 2, 8)

    imshow(drawing, 'drawing')

    final_bboxes = []
    final_coutours = []
    # print(bboxes)
    for i, c in enumerate(new_contours):
        box = bboxes[i]
        box_width = box[2]
        box_height = box[3]
        area = box_height * box_width

        if (box_width > max_width and box_height > max_height):
            # print(max_height, max_width)
            # print(box_height, box_width)
            # print('contour removed due to size')
            pass
        elif area < mean_area:
            # print('contour removed due to area')
            pass
        elif c.shape[0] != 4:
            # print('contour removed due to shape')
            pass
        else:
            final_bboxes.append(box)
            final_coutours.append(c)

    if len(final_coutours) == 0:
        print('all contours removed!')
        return None

    if len(final_coutours) > 1:
        final_coutours = sorted(final_coutours, key=cv2.contourArea, reverse=True)
        final_bboxes = sorted(final_bboxes, key=calc_bbox_area, reverse=True)

    board_contour = final_coutours[0]
    board_box = final_bboxes[0]

    if board_contour.shape[0] != 4:
        print('final contour errer!')
        return None

    center = (board_box[0] + board_box[2] / 2, board_box[1] + board_box[3] / 2)

    # print(board_contour)
    upper = []
    lower = []
    for p in board_contour[:, 0, :]:
        if p[1] < center[1]:
            upper.append(p)
        else:
            lower.append(p)

    if len(upper) != 2 or len(lower) != 2:
        print('can not split')
        return None

    p0 = upper[0]
    p1 = upper[1]
    if p0[0] > p1[0]:
        p0 = upper[1]
        p1 = upper[0]

    p2 = lower[0]
    p3 = lower[1]
    if p2[0] < p3[0]:
        p2 = lower[1]
        p3 = lower[0]

    final_draw = img.copy()
    cv2.rectangle(final_draw, (board_box[0], board_box[1]),
                  (board_box[0] + board_box[2], board_box[1] + board_box[3]),
                  (0, 0, 255), 4, 8)
    cv2.drawContours(final_draw, [board_contour], 0, (0, 255, 0), 2, 8)

    print('Auto detect corner:', p0, p1, p2, p3)

    imshow(final_draw, 'final_draw')

    return p0, p1, p2, p3


def getWarpedImg(img):
    corner = automatic_warp(img)

    if corner is None:
        raise ValueError('Failed to found board!')

    warpedImg = warpImg(img, corner)

    return warpedImg


def warpImg(img, corner):
    imgSize = min(img.shape[0:2])
    dstSize = imgSize

    src = []
    for i in range(4):
        src.append(corner[i])

    dst = [[0, 0], [dstSize, 0], [dstSize, dstSize], [0, dstSize]]

    src, dst = np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

    # print(src)
    # print(dst)

    m = cv2.getPerspectiveTransform(src, dst)

    warpedImg = cv2.warpPerspective(img, m, (imgSize, imgSize), borderValue=(255, 255, 255))

    return warpedImg
