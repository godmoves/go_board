import cv2
import numpy as np
import matplotlib.pyplot as plt


def distance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def find_corner(point, ref):
    x_min = ref[0]
    x_max = ref[1]
    y_min = ref[2]
    y_max = ref[3]
    dist_left_up = [distance(p, (x_min, y_max)) for p in point]
    dist_left_down = [distance(p, (x_min, y_min)) for p in point]
    dist_right_up = [distance(p, (x_max, y_max)) for p in point]
    dist_right_down = [distance(p, (x_max, y_min)) for p in point]

    left_up = point[np.argmin(dist_left_up)]
    left_down = point[np.argmin(dist_left_down)]
    right_up = point[np.argmin(dist_right_up)]
    right_down = point[np.argmin(dist_right_down)]

    return left_down, left_up, right_down, right_up


def delete_small_component(img):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 150

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2


def auto_correction(figure_name, more_info=False):
    img = cv2.imread(figure_name)
    print('Image shape:', img.shape)
    w, h = img.shape[0:2]
    size = min(w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(gray)
    # plt.show()

    edges = cv2.Canny(gray, 45, 200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    img2 = delete_small_component(edges)

    edges3 = img2.astype(int)

    if more_info:
        plt.subplot(221)
        plt.imshow(edges)
        plt.subplot(222)
        plt.imshow(edges3)
        plt.subplot(223)
        plt.imshow(edges - edges3)
        plt.show()

    # for i in range(w):
    #     for j in range(h):
    #         edges[i, j] = edges3[i, j]

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            maxLineGap=10, minLineLength=int(0.1 * size))
    lines1 = lines[:, 0, :]
    x = []
    y = []
    point = []
    for x1, y1, x2, y2 in lines1[:]:
        x.append(x1)
        x.append(x2)
        y.append(y1)
        y.append(y2)
        point.append([x1, y1])
        point.append([x2, y2])
        if more_info:
            cv2.line(edges_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.circle(edges_bgr, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(edges_bgr, (x2, y2), 5, (255, 0, 0), -1)

    ref = [np.min(x), np.max(x), np.min(y), np.max(y)]
    ld, lu, rd, ru = find_corner(point, ref)
    if more_info:
        cv2.circle(edges_bgr, tuple(ld), 10, (0, 255, 0), -1)
        cv2.circle(edges_bgr, tuple(lu), 10, (0, 255, 0), -1)
        cv2.circle(edges_bgr, tuple(rd), 10, (0, 255, 0), -1)
        cv2.circle(edges_bgr, tuple(ru), 10, (0, 255, 0), -1)

    t_min = int(0.05 * size)
    t_max = int(0.95 * size)
    src = np.array([ld, lu, rd, ru], dtype=np.float32)
    dst = np.array([[t_min, t_min], [t_min, t_max], [
                   t_max, t_min], [t_max, t_max]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    new_point = cv2.perspectiveTransform(
        np.array([point], dtype=np.float32), m)
    img = cv2.warpPerspective(img, m, (size, size), borderValue=(255, 255, 255))
    target_img = img.copy()

    x = []
    y = []
    point = []
    for p in new_point[0]:
        x.append(p[0])
        y.append(p[1])
        point.append(p)
        # if more_info:
        #     cv2.circle(img, (p[0], p[1]), 5, (0, 255, 0), -1)

    ref = [np.min(x), np.max(x), np.min(y), np.max(y)]
    ld, lu, rd, ru = find_corner(point, ref)

    src = np.array([ld, lu, rd, ru], dtype=np.float32)
    dst = np.array([[t_min, t_min], [t_min, t_max], [
                   t_max, t_min], [t_max, t_max]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, m, (size, size), borderValue=(255, 255, 255))
    target_img = cv2.warpPerspective(target_img, m, (size, size), borderValue=(255, 255, 255))

    if more_info:
        cv2.circle(img, (t_min, t_min), 10, (0, 0, 255), -1)
        cv2.circle(img, (t_min, t_max), 10, (0, 0, 255), -1)
        cv2.circle(img, (t_max, t_min), 10, (0, 0, 255), -1)
        cv2.circle(img, (t_max, t_max), 10, (0, 0, 255), -1)
        cv2.line(img, (t_min, t_min), (t_min, t_max), (0, 0, 255), 3)
        cv2.line(img, (t_min, t_max), (t_max, t_max), (0, 0, 255), 3)
        cv2.line(img, (t_max, t_max), (t_max, t_min), (0, 0, 255), 3)
        cv2.line(img, (t_max, t_min), (t_min, t_min), (0, 0, 255), 3)

        plt.subplot(121)
        plt.imshow(edges_bgr, cmap='gray')
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.show()

    return target_img


def main():
    figure_name = './TestImage02.jpg'
    auto_correction(figure_name, more_info=True)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    # plt.imshow(img, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main()
