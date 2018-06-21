import cv2
import numpy as np
import random

from utils import imshow


class PartitionOperator():
    def __init__(self, size, board_size):
        self.size = size
        if board_size == 0:
            self.board_size = random.choice([9, 13, 19])
        else:
            self.board_size = board_size
        self.boardfield = ((1.0 / self.board_size) * self.size) / 3.0

    def compare(self, a, b):
        distance = abs(a - b)
        return distance < self.boardfield


def partition(points, Oper):
    clusterSize = 0
    clusterNum = []
    pos_now = points[0]
    for p in points:
        if Oper.compare(pos_now, p):
            clusterNum.append(clusterSize)
        else:
            clusterSize += 1
            clusterNum.append(clusterSize)
            pos_now = p
    if len(points) != len(clusterNum):
        raise ValueError('Missing cluster points')
    return clusterSize, clusterNum


def createLinefromValue(circles, line_type, board_size, imgheight, imgwidth):
    if line_type == 'VERTICAL':
        valueIndex1 = 0
        valueIndex2 = 2
        imagesizeIndex = 3
        zeroIndex = 1
        imagesize = imgheight
    elif line_type == 'HORIZONTAL':
        valueIndex1 = 1
        valueIndex2 = 3
        imagesizeIndex = 2
        zeroIndex = 0
        imagesize = imgwidth

    Oper = PartitionOperator(imagesize, board_size)
    clusterSize, clusterNum = partition(circles, Oper)

    cluster_we_need = []
    new_clusterSize = 0
    for i in range(clusterSize):
        num_of_one_cluster = 0
        for k in clusterNum:
            if k == i:
                num_of_one_cluster += 1
            if num_of_one_cluster > 3:  # tune this to find more lines
                cluster_we_need.append(i)
                new_clusterSize += 1
                break

    clusterdCircles = []
    for k in range(new_clusterSize):
        temp = []
        for i, c in enumerate(clusterNum):
            if c == cluster_we_need[k]:
                temp.append(circles[i])
        clusterdCircles.append(temp)

    if len(clusterdCircles) != new_clusterSize:
        raise ValueError('Not equal cluster size')

    middle = [0, 0, 0, 0]
    middleLines = []

    for i in range(new_clusterSize):
        mid = np.mean(clusterdCircles[i])

        middle[zeroIndex] = 0
        middle[valueIndex1] = mid
        middle[imagesizeIndex] = imagesize
        middle[valueIndex2] = mid

        middleLines.append(middle.copy())

    return middleLines


def getBetterDetectionImage(houghImg, createFakeLines, board_size):
    houghCircleImg = houghImg.copy()
    element_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    houghCircleImg = cv2.dilate(houghCircleImg, element_dilate)

    # imshow(houghCircleImg, 'houghCircleImg')

    if board_size == 0:
        min_radius = 20
        max_radius = 30
    else:
        min_radius = int((1.0 / board_size) * 200.0 - 4.0)
        max_radius = int((1.0 / board_size) * 320.0 + 4.0)

    circles = cv2.HoughCircles(houghCircleImg, cv2.HOUGH_GRADIENT, 1, 30,
                               param1=30, param2=10, minRadius=min_radius,
                               maxRadius=max_radius)
    circles = circles[0, :, :]

    houghCircleImg2 = cv2.cvtColor(houghCircleImg, cv2.COLOR_GRAY2RGB)
    for i in circles[:]:
        cv2.circle(houghImg, (i[0], i[1]), int(i[2] + 5), 0, -1, 8, 0)
        cv2.circle(houghCircleImg2, (i[0], i[1]), int(i[2] + 5), (0, 0, 255), -1, 8, 0)

    # imshow(houghImg, 'houghImg')
    # imshow(houghCircleImg2, 'houghCircleImg2')

    if createFakeLines:
        circles_x = []
        circles_y = []
        for i in circles:
            circles_x.append(i[0])
            circles_y.append(i[1])
        sorted_x = sorted(circles_x)
        sorted_y = sorted(circles_y)

        if len(sorted_x) != 0:
            vlines = createLinefromValue(sorted_x, 'VERTICAL', board_size,
                                         houghImg.shape[0], houghImg.shape[1])
        if len(sorted_y) != 0:
            hlines = createLinefromValue(sorted_y, 'HORIZONTAL', board_size,
                                         houghImg.shape[0], houghImg.shape[1])

        newLines = vlines + hlines

        houghImg2 = houghImg.copy()
        houghImg2 = cv2.cvtColor(houghImg2, cv2.COLOR_GRAY2RGB)
        for i, l in enumerate(newLines):
            # print('line {} from ({:.2f},{:.2f}) to ({:.2f},{:.2f})'.format(
            #     i, l[0], l[1], l[2], l[3]))
            cv2.line(houghImg, (l[0], l[1]), (l[2], l[3]), 255, 1, 8)
            cv2.line(houghImg2, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, 8)

    # imshow(houghImg, 'hough test')
    # imshow(houghImg2, 'hough2 test')

    return houghImg


def groupIntersectionLines(lines, imgheight, imgwidth):
    hlines = []
    vlines = []

    baseVector = [imgwidth, 0]

    for l in lines:
        lineVector = [l[2] - l[0], l[3] - l[1]]
        angle = calcBetweenAngle(baseVector, lineVector)

        if (angle <= 1 and angle >= -1):
            if ((l[0] - l[2]) != 0):
                m = (l[1] - l[3]) / (l[0] - l[2])

                l2 = np.zeros(4)
                l2[0] = 0
                l2[1] = int(-m * l[0] + l[1])
                l2[2] = imgwidth
                l2[3] = int(m * (imgwidth - l[2]) + l[3])
            else:
                l2 = np.zeros(4)
                l2[0] = 0
                l2[1] = l[1]
                l2[2] = imgwidth
                l2[3] = l[3]

            hlines.append(l2.copy())

        elif (angle >= 88 and angle <= 92):
            if((l[0] - l[2]) != 0):
                m = (l[1] - l[3]) / (l[0] - l[2])
                n = l[1] - m * l[0]

                l2 = np.zeros(4)
                l2[0] = int(-n / m)
                l2[1] = 0
                l2[2] = int((imgheight - n) / m)
                l2[3] = imgheight
            else:
                l2 = np.zeros(4)
                l2[0] = l[0]
                l2[1] = 0
                l2[2] = l[2]
                l2[3] = imgheight
            vlines.append(l2.copy())
        else:
            pass  # we just ignore these lines that neither horizontal nor vertical
    return hlines, vlines


def calcBetweenAngle(v1, v2):
    l1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    l2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
    inner_product = v1[0] * v2[0] + v1[1] * v2[1]

    angle = np.arccos(inner_product / (l1 * l2))
    angle = angle * (180.0 / np.pi)
    return angle


def getBoardLines(lines, line_type, board_size, imgheight, imgwidth):
    if line_type == 'VERTICAL':
        valueIndex1 = 0
        valueIndex2 = 2
        imagesizeIndex = 3
        zeroIndex = 1
        imagesize = imgheight
    elif line_type == 'HORIZONTAL':
        valueIndex1 = 1
        valueIndex2 = 3
        imagesizeIndex = 2
        zeroIndex = 0
        imagesize = imgwidth

    lineStarts = []
    lineEnds = []
    for l in lines:
        lineStarts.append(l[valueIndex1])
        lineEnds.append(l[valueIndex2])

    Oper = PartitionOperator(imagesize, board_size)
    clusterSize, clusterNum = partition(lineStarts, Oper)

    clusterLinseStarts = []
    clusterLineEnds = []

    for j in range(clusterSize):
        temp_starts = []
        temp_ends = []
        for i in range(len(clusterNum)):
            if clusterNum[i] == j:
                temp_starts.append(lineStarts[i])
                temp_ends.append(lineEnds[i])
        clusterLinseStarts.append(temp_starts)
        clusterLineEnds.append(temp_ends)

    middle = [0, 0, 0, 0]
    middleLines = []

    for i in range(clusterSize):
        midStarts = np.mean(clusterLinseStarts[i])
        midEnds = np.mean(clusterLineEnds[i])

        middle[zeroIndex] = 0
        middle[valueIndex1] = midStarts
        middle[imagesizeIndex] = imagesize
        middle[valueIndex2] = midEnds

        middleLines.append(middle.copy())

    return middleLines


def intersection(h, v, imgheight, imgwidth):
    o1 = np.array([h[0], h[1]])
    p1 = np.array([h[2], h[3]])
    o2 = np.array([v[0], v[1]])
    p2 = np.array([v[2], v[3]])

    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return False, None

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = o1 + d1 * t1
    if r[0] >= imgwidth or r[1] >= imgheight:
        return False, None

    return True, r


def getBoardIntersections(warpedImg, thresholdValue, board_size):
    imgheight = warpedImg.shape[0]
    imgwidth = warpedImg.shape[1]

    maxValue = 255
    thresholdType = 4

    warpedImgGray = cv2.cvtColor(warpedImg, cv2.COLOR_BGR2GRAY)

    cannyImg = cv2.Canny(warpedImgGray, 100, 150, 3)

    # Not useful?
    thresholdImg = cv2.threshold(cannyImg, 255, maxValue, thresholdType)[1]

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresholdImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, element)

    thresholdImg = getBetterDetectionImage(thresholdImg, True, board_size)

    # imshow(thresholdImg, 'thresholdImg for HoughLinesP')

    lines = cv2.HoughLinesP(thresholdImg, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=10)
    lines = lines[:, 0, :]

    houghImg = warpedImg.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(houghImg, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)

    # imshow(houghImg, 'HoughLines Image')

    hlines, vlines = groupIntersectionLines(lines, imgheight, imgwidth)

    if len(hlines) != 0:
        new_hlines = getBoardLines(hlines, 'HORIZONTAL', board_size, imgheight, imgwidth)
    if len(vlines) != 0:
        new_vlines = getBoardLines(vlines, 'VERTICAL', board_size, imgheight, imgwidth)

    intersectionPonits = []
    for h in new_hlines:
        for v in new_vlines:
            result, P = intersection(h, v, imgheight, imgwidth)
            if result:
                intersectionPonits.append(P)

    newLines = new_hlines + new_vlines

    paintedImg = warpedImg.copy()
    for l in newLines:
        cv2.line(paintedImg, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 1, 8)

    for p in intersectionPonits:
        x1 = int(p[0] - 1)
        y1 = int(p[1] - 1)
        x2 = int(p[0] + 1)
        y2 = int(p[1] + 1)
        cv2.rectangle(paintedImg, (x1, y1), (x2, y2), (0, 255, 0), 2, 8, 0)

    print('{} lines, {} intersection points'.format(len(newLines), len(intersectionPonits)))

    return paintedImg, intersectionPonits
