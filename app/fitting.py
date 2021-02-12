import cv2
import numpy as np
from scipy.optimize import minimize, line_search


def ellipse(image, x0 = 0, y0 = 0, r = 100):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image.clone()

    height, width = gray.shape

    points = []

    for y in range(0, height - 1):
        for x in range(0, width - 1):
            if gray[y, x] > 100:
                points.append((x, y))

    a = 1
    b = 0
    c = 1
    d = - 2*x0
    e = - 2*y0
    f = y0**2 + x0**2 - r**2

    start_point = np.array([a, b, c, d, e, f])

    def equation(p, x, y):
        return p[0] * x**2 + p[1] * x * y + p[2] * y**2 + p[3] * x + p[4] * y + p[5]

    def obj_func(p):
        acc = 0

        for x, y in points:
            acc += (equation(p, x, y))**2

        return acc

    res = minimize(obj_func, start_point)

    kernel = np.zeros((height, width), np.float32)

    for y in range(0, height):
        for x in range(0, width):
            kernel[(y, x)] += equation(res.x, x, y)

    return kernel
