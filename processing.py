import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_2D(image, cmap="gray", title=""):
    plt.title(title)
    plt.imshow(np.squeeze(image), cmap=cmap)
    plt.show()

def generate_side_coords(h_offset, l_offset, derivative, offset=100):
    """
    create a set of points orthogonal to the curve
    Args:
        h_offset (3D numpy array): coords for the higher offset foreach point of the curve
        l_offset (3D numpy array): coords for lower offset for each point of the curve
        derivative (2D numpy array): first derivative for each point of the curve
        offset (Int): space between offsets (used for calculating the steps)

    Returns:
    points (ndarray) a set of (y,x) coords for each line.
    """
    slices = []
    for (x1, y1), (x2, y2), alfa in zip(h_offset, l_offset, derivative):
        sign = 1 if alfa > 0 else -1
        x_step = abs(x1 - x2) / offset
        y_step = abs(y1 - y2) / offset
        points = [
            [(x1 + sign * i * x_step), (y1 + (i * y_step))] for i in
            range(offset + 1)
        ]
        slices.append(points)
    return np.array(slices)


def compute_skeleton(img):
    """
    create the skeleton using morphology
    Args:
        img (numpy array): source image

    Returns:
        (numpy array), b&w image: 0 background, 255 skeleton elements
    """

    img = img.astype(np.uint8)
    size = img.size
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = np.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            return skel


def arch_detection(slice, debug=False):
    """
    compute a polynomial spline of the dental arch from a DICOM file
    Args:
        slice (numpy array): source image. Must be float with values in range [0,1]
        debug (Bool): if true the result is shown at each step

    Returns:
        (poly1d object): polynomial function approximation
        (float) starting value for the X axis
        (float) ending value for the X axis
    """

    def score_func(arch, th):
        tmp = cv2.threshold(arch, th, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        score = tmp[tmp == 1].size / tmp.size
        return score

    if debug:
        plot_2D(slice)
    # initial closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    arch = cv2.morphologyEx(slice, cv2.MORPH_CLOSE, kernel)

    th = 0.50

    max_it = 20
    step_l = 0.01

    # varying threshold until we white area is about 12%
    h_th = 0.17
    l_th = 0.11

    poly_x = [th / 20 for th in range(0, 20, 1)]
    poly_y = [score_func(arch, th) for th in poly_x]
    th2score = np.poly1d(np.polyfit(poly_x, poly_y, 12))

    for _ in range(max_it):
        score = th2score(th)
        if h_th > score > l_th:
            arch = cv2.threshold(arch, th, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
            break
        d = -np.polyder(th2score)(th)
        if score > h_th:
            th = th + step_l * d  # higher threshold, lower score
        elif score < l_th:
            th = th - step_l * d  # lower threshold, higher score

        debug and plot_2D(cv2.threshold(arch, th, 1, cv2.THRESH_BINARY)[1].astype(np.uint8),
                                 title="th: {} - score: {}".format(th, score))

    # hole filling
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # arch = cv2.morphologyEx(arch, cv2.MORPH_CLOSE, kernel)
    # if debug:
    #     plot_2D(arch)

    # major filtering with labelling: first remove all the little white parts
    ret, labels = cv2.connectedComponents(arch)
    sizes = np.asarray([labels[labels == label].size for label in range(1, ret)])
    labels[labels != (sizes.argmax() + 1)] = 0  # set all not maximum components to background
    labels[labels == (sizes.argmax() + 1)] = 1  # set the biggest components as foreground

    # let's now fill the rest of the holes if any
    labels = 1 - labels
    ret, labels = cv2.connectedComponents(labels.astype(np.uint8))
    sizes = np.asarray([labels[labels == label].size for label in range(1, ret)])
    labels[labels != (sizes.argmax() + 1)] = 0
    labels[labels == (sizes.argmax() + 1)] = 1
    labels = 1 - labels

    # for label in range(1, ret):
    #     if labels[labels == label].size < 10000:
    #         labels[labels == label] = 0
    if debug:
        plot_2D(labels)

    # compute skeleton
    skel = compute_skeleton(labels)
    if debug:
        plot_2D(skel)

    # # labelling on the resulting skeleton
    # cs, im = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filtered = []
    # for c in cs:
    #     if len(c) > 40:
    #         filtered.append(c)
    #
    # # creating the mask of chosen pixels
    # contour = np.zeros(skel.shape, np.uint8)
    # cv2.drawContours(contour, filtered, -1, 255)
    # if debug:
    #     plot_2D(contour)

    # regression polynomial function
    coords = np.argwhere(skel > 0)
    y = [y for y, x in coords]
    x = [x for y, x in coords]
    pol = np.polyfit(x, y, 12)
    p = np.poly1d(pol)

    # generating the curve for a check
    if debug:
        recon = np.zeros(skel.shape, np.uint8)  # binary image for test
        original_rgb = np.tile(slice, (3, 1, 1))  # overlay on the original image (colorful)
        original_rgb = np.moveaxis(original_rgb, 0, -1)
        for sample in np.linspace(min(x), max(x), 1000):  # range(min(x), max(x)):
            y_sample = p(sample)
            recon[int(y_sample), int(sample)] = 255
            original_rgb[int(y_sample), int(sample), :] = (255, 0, 0)
        plot_2D(recon)
        plot_2D(original_rgb, cmap=None)

    return p, min(x), max(x)


def arch_lines(func, start, end, offset=50, d=1):
    """
    this functions uses the first order derivative of the function func to track the proper points (x,y) from start to end.
    Args:
        func (poly1d object): polynomial function approximation
        end (float) starting value for the X axis
        start (float) ending value for the X axis
        offset (Int): offset for generating two more curves

    Returns:
        low_offset (numpy array): set of sets of xy coordinates (lower offset)
        coords (numpy array): set of sets of xy coordinates
        high_offset (numpy array): set of sets of xy coordinates (higer offset)
        derivative: set of derivates foreach point of coords
    """
    delta = 0.3
    coords = []
    x = start + 1
    # we start from the range of X values on the X axis,
    # we create a new list X of x coords along the curve
    # we exploit the first order derivative to place values in X
    # so that f(X) is equally distant for each point in X
    while x < end:
        coords.append((x, func(x)))
        alfa = (func(x + delta / 2) - func(x - delta / 2)) / delta
        x = x + d * np.sqrt(1 / (alfa ** 2 + 1))

    # creating lines parallel to the spline
    high_offset = []
    low_offset = []
    derivative = []
    for x, y in coords:
        alfa = (func(x + delta / 2) - func(x - delta / 2)) / delta  # first derivative
        alfa = -1 / alfa  # perpendicular coeff
        cos = np.sqrt(1 / (alfa ** 2 + 1))
        sin = np.sqrt(alfa ** 2 / (alfa ** 2 + 1))
        if alfa > 0:
            low_offset.append((x + offset * cos, y + offset * sin))
            high_offset.append((x - offset * cos, y - offset * sin))
        else:
            low_offset.append((x - offset * cos, y + offset * sin))
            high_offset.append((x + offset * cos, y - offset * sin))
        derivative.append(alfa)

    return low_offset, coords, high_offset, derivative


def increase_contrast(image):
    """
    increase the contrast of an image using https://www.sciencedirect.com/science/article/pii/B9780123361561500616
    Args:
        image (numpy array): 0-1 floating image
    Returns:
        result (numpy array): image with higer contrast
    """
    if image.max() > 1 or image.min() < 0:
        raise Exception("increase contrast: input image should have values between 0 and 1")
    sharp = image * 255  # [0 - 255]
    sharp = sharp.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    sharp = clahe.apply(sharp)
    sharp = sharp.astype(np.float32) / 255  # back to [0-1]
    return sharp


def grey_to_rgb(grey):
    """
    create a color image/volume from its grey scale version
    Args:
        grey (numpy array): grey scale image
    Returns:
        result (numpy array): RGB volume or image
    """
    if len(grey.shape) == 3:
        rgb = np.tile(grey, (3, 1, 1, 1))  # volume
    else:
        rgb = np.tile(grey, (3, 1, 1))  # image
    return np.moveaxis(rgb, 0, -1)


def angle_from_centroids(pred_0, pred_1, plane_0, plane_1):
    """
    compute the angle for a proper inclination given two binary slices of predictions/annotations.
    inclination is given by the angles to use for tilting the next cutting plane around the z and x axis for
    a achieving a better section
    Args:
        pred_0 (np.array): binary numpy array where 1 means canal is present at that pixel
        pred_1 (np.array): same as pred_0
        plane_0 (Plane obj): plane with xyz coords used for generating the section of pred_0
        plane_1 (Plane obj): plane with xyz coords used for generating the section of pred_1

    Returns:
        angle_z (float)
        angle_x (float)
    """
    if np.any(pred_0 != 0) and np.any(pred_1 != 0):
        true_coords = np.argwhere(pred_0)
        xyz_0 = plane_0[:, true_coords[:, 0], true_coords[:, 1]]
        centroid_0 = xyz_0.mean(axis=1)

        true_coords = np.argwhere(pred_1)
        xyz_1 = plane_1[:, true_coords[:, 0], true_coords[:, 1]]
        centroid_1 = xyz_1.mean(axis=1)

        diff = centroid_1 - centroid_0  # 2 -> z, 1 -> y, 0 -> x
        diff[2] = - diff[2]

        z_angle = np.degrees(np.arctan(diff[2] / diff[0]))  # z / x
        x_angle = np.degrees(np.arctan(diff[1] / diff[0]))  # y / x
        return z_angle, x_angle
    else:
        return 0, 0  # cant compute centroids from black masks
