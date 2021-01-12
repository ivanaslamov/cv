import cv2
import numpy as np

from app.utils import gaussian, gradient, hessian_determinant


def moravec(image):
    window_size = 4

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image.clone()

    height, width = gray.shape

    mat = np.zeros((height, width), np.float32)
    maximum = 0

    for y in range(window_size, height - 3*window_size):
        for x in range(window_size, width - 3*window_size):
            # patch = gray[y - window_size:y + window_size, ]
            # patch = gray[y - window_size:y + window_size, x - window_size:x + window_size]
            patch = gray[y:y + 2*window_size, x:x + 2*window_size]

            acc = None

            for u in [y-window_size, y+window_size]:
                for v in [x-window_size, x+window_size]:
                    uv_patch = gray[u:u + 2*window_size, v:v + 2*window_size]

                    if acc is None:
                        acc = np.sum((patch - uv_patch)**2)
                    else:
                        acc = min(acc, np.sum((patch - uv_patch)**2))

            maximum = max(maximum, acc)
            mat[(y, x)] = acc

    mat /= maximum

    return mat


def shi(image):
    window_size = 8

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image.clone()

    height, width = gray.shape

    ones_helper = gaussian(window_size)

    k = 0.05
    thresh = 1000000

    # algorithm
    dy, dx = gradient(gray)

    i_xx = dx ** 2
    i_xy = dy * dx
    i_yy = dy ** 2

    cxx = cv2.filter2D(i_xx, -1, ones_helper)
    cxy = cv2.filter2D(i_xy, -1, ones_helper)
    cyy = cv2.filter2D(i_yy, -1, ones_helper)

    det = (cxx * cyy) - (cxy ** 2)
    trace = cxx + cyy
    r = det - k * (trace ** 2)

    for y in range(0, height):
        for x in range(0, width):
            if r[(y, x)] > thresh:
                cv2.circle(image, (x, y), 3, 255, -1)

    return image


def eigen(image):
    window_size = 8

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image.clone()

    height, width = gray.shape

    ones_helper = gaussian(window_size)

    thresh = 100

    # algorithm
    dy, dx = gradient(gray)

    i_xx = dx ** 2
    i_xy = dy * dx
    i_yy = dy ** 2

    cxx = cv2.filter2D(i_xx, -1, ones_helper)
    cxy = cv2.filter2D(i_xy, -1, ones_helper)
    cyy = cv2.filter2D(i_yy, -1, ones_helper)

    corners = np.zeros((height, width), np.float32)

    for y in range(0, height):
        print(y)
        for x in range(0, width):
            cov = np.array([[cxx[(y, x)], cxy[(y, x)]], [cxy[(y, x)], cyy[(y, x)]]], np.float32)
            evals, _ = np.linalg.eig(cov)

            if len(evals) == 2 and evals[0] > 0 and evals[1] > 0:
                lower_radius = min(evals[0], evals[1])
                corners[(y, x)] = lower_radius

    # non-maxima suppression (still need to find
    dxx, dyy, dxy, det = hessian_determinant(corners)

    for y in range(0, height):
        for x in range(0, width):
            if det[(y, x)] > 0 and dxx[(y, x)] > 0 and corners[(y, x)] > thresh:
                cv2.circle(image, (x, y), 3, 255, -1)

    return image
