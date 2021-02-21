import cv2
import numpy as np


def gaussian(n):
    s = 0.3 * (n / 2 - 1) + 0.8

    mat = np.zeros((n, n), np.float32)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = np.power(x - n / 2.0, 2) + np.power(y - n / 2.0, 2)
            mat[(x, y)] = 1 / (2.0 * np.pi * np.power(s, 2)) * np.exp(- sq_dist / 2.0 / s / s)

    return mat[1:, 1:]


def gaussian_derivative_x(n):
    s = 0.3 * (n / 2 - 1) + 0.8

    mat = np.zeros((n, n), np.float32)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = np.power(x - n / 2.0, 2) + np.power(y - n / 2.0, 2)
            mat[(y, x)] = (x - n / 2.0) / (2 * np.pi * np.power(s, 4)) * np.exp(- sq_dist / (2.0 * np.power(s, 2)))

    return mat[1:, 1:]


def gaussian_derivative_xx(n):
    s = 0.3 * (n / 2 - 1) + 0.8

    mat = np.zeros((n, n), np.float32)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = np.power(x - n / 2.0, 2) + np.power(y - n / 2.0, 2)
            mat[(y, x)] = (1 - np.power(x - n / 2.0, 2) / np.power(s, 2)) / (2 * np.pi * np.power(s, 4)) * np.exp(
                - sq_dist / (2.0 * np.power(s, 2)))

    return mat[1:, 1:]


def gaussian_derivative_y(n):
    s = 0.3 * (n / 2 - 1) + 0.8

    mat = np.zeros((n, n), np.float32)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = np.power(x - n / 2.0, 2) + np.power(y - n / 2.0, 2)
            mat[(y, x)] = (y - n / 2.0) / (2 * np.pi * np.power(s, 4)) * np.exp(- sq_dist / (2.0 * np.power(s, 2)))

    return mat[1:, 1:]


def gaussian_derivative_yy(n):
    s = 0.3 * (n / 2 - 1) + 0.8

    mat = np.zeros((n, n), np.float32)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = np.power(x - n / 2.0, 2) + np.power(y - n / 2.0, 2)
            mat[(y, x)] = (1 - np.power(y - n / 2.0, 2) / np.power(s, 2)) / (2 * np.pi * np.power(s, 4)) * np.exp(
                - sq_dist / (2.0 * np.power(s, 2)))

    return mat[1:, 1:]


def gaussian_derivative_xy(n):
    s = 0.3 * (n / 2 - 1) + 0.8

    mat = np.zeros((n, n), np.float32)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = np.power(x - n / 2.0, 2) + np.power(y - n / 2.0, 2)
            mat[(y, x)] = \
                - (x - n / 2.0) * (y - n / 2.0) / (2 * np.pi * np.power(s, 6)) * \
                np.exp(- sq_dist / (2.0 * np.power(s, 2)))

    return mat[1:, 1:]


def gaussian_derivative_phi(phi, n):
    s = 0.3 * (n / 2 - 1) + 0.8

    theta = 90

    mx = np.sin(theta * np.pi / 180) * np.cos(phi * np.pi / 180)
    my = np.sin(theta * np.pi / 180) * np.sin(phi * np.pi / 180)

    magnitude = np.sqrt(mx * mx + my * my)
    ux = mx / magnitude;
    uy = my / magnitude;

    mat = np.zeros((n, n), np.float32)
    c = 1 / 2.0 / np.power(s, 4)

    for x in range(0, n):
        for y in range(0, n):
            sq_dist = (x - n / 2.0) * (x - n / 2.0) + (y - n / 2.0) * (y - n / 2.0)
            dx = c * (x - n / 2.0) / s / s * np.exp(- sq_dist / 2.0 / s / s)
            dy = c * (y - n / 2.0) / s / s * np.exp(- sq_dist / 2.0 / s / s)

            mat[(y, x)] = dx * ux + dy * uy

    return mat[1:, 1:]


def gradient(img, kernel_size=9):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if len(img.shape) == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    if img.dtype != 'float32':
        img = img.astype('float32')

    kernel_x = gaussian_derivative_x(kernel_size)
    kernel_y = gaussian_derivative_y(kernel_size)
    dx = cv2.filter2D(img, -1, kernel_x)
    dy = cv2.filter2D(img, -1, kernel_y)

    return dy, dx


def gradient_magnitude(img):
    dy, dx = gradient(img)

    return np.sqrt(dy * dy + dx * dx)


def hessian_determinant(gray):
    kernel_size = 8

    kernel_xx = gaussian_derivative_xx(kernel_size)
    kernel_yy = gaussian_derivative_yy(kernel_size)
    kernel_xy = gaussian_derivative_xy(kernel_size)
    dxx = cv2.filter2D(gray, -1, kernel_xx)
    dyy = cv2.filter2D(gray, -1, kernel_yy)
    dxy = cv2.filter2D(gray, -1, kernel_xy)

    det = (dxx * dyy) - (dxy ** 2)

    return dxx, dyy, dxy, det


def ones(n):
    return np.ones((n, n), np.float32)


def preview_gaussian():
    kernel = gaussian(256)
    kernel *= 10000
    kernel += .5

    cv2.imshow('frame', kernel)
    cv2.waitKey()


def preview_gaussian_derivative_x():
    kernel = gaussian_derivative_x(256)
    kernel *= 1000000
    kernel += .5

    cv2.imshow('frame', kernel)
    cv2.waitKey()


def preview_gaussian_derivative_y():
    kernel = gaussian_derivative_y(256)
    kernel *= 1000000
    kernel += .5

    cv2.imshow('frame', kernel)
    cv2.waitKey()


def preview_gaussian_derivative_xx():
    kernel = gaussian_derivative_xx(256)
    kernel *= 10000000
    kernel += .5

    cv2.imshow('frame', kernel)
    cv2.waitKey()


def preview_gaussian_derivative_yy():
    kernel = gaussian_derivative_yy(256)
    kernel *= 10000000
    kernel += .5

    cv2.imshow('frame', kernel)
    cv2.waitKey()


def preview_gaussian_derivative_xy():
    kernel = gaussian_derivative_xy(256)
    kernel *= 10000000
    kernel += .5

    cv2.imshow('frame', kernel)
    cv2.waitKey()


def preview_gaussian_derivative_phi():
    angle = 0
    while True:
        kernel = gaussian_derivative_phi(angle, 256)
        kernel *= 100000000
        kernel += .5

        cv2.imshow('frame', kernel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        angle += 1


def distance(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)


def curvature(x0, y0, x1, y1, x2, y2):
    return np.sqrt((x0-2*x1+x2)**2 + (y0-2*y1+y2)**2)
