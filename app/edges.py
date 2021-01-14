import cv2
import numpy as np

from app.utils import gradient


def canny(image, threshold_one, threshold_two):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image.clone()

    height, width = gray.shape

    magnitudes = np.zeros((height, width), np.float32)

    dy, dx = gradient(gray)

    maximum = 0

    for y in range(0, height):
        for x in range(0, width):
            magnitude = np.sqrt(np.square(dx[(y, x)]) + np.square(dy[(y, x)]))
            magnitudes[(y, x)] = magnitude
            maximum = max(maximum, magnitude)

    edges = np.zeros((height, width), np.float32)

    # non maxima supression

    for y in range(1, height-1):
        for x in range(1, width-1):
            magnitude = magnitudes[(y, x)]

            if magnitude < 0.001:
                continue

            norm_x = dx[(y, x)] / magnitude
            norm_y = dy[(y, x)] / magnitude

            y_minus = int(np.floor(y + 0.5 - norm_y))
            x_minus = int(np.floor(x + 0.5 - norm_x))

            y_plus = int(np.floor(y + 0.5 + norm_y))
            x_plus = int(np.floor(x + 0.5 + norm_x))

            pixel_one = magnitudes[(y_minus, x_minus)]
            pixel_two = magnitudes[(y_plus, x_plus)]

            if magnitude > pixel_one and magnitude > pixel_two:
                edges[(y, x)] = magnitude

    # histerisis thresholding
    pixels = []

    for y in range(0, height):
        for x in range(0, width):
            value = edges[(y, x)]

            if value > threshold_one:
                pixels.append( (y, x) )

    output = np.zeros((height, width), np.float32)

    while pixels:
        y, x = pixels.pop()

        if output[(y, x)]:
            continue

        output[(y, x)] = 1

        cords = [
            (y - 1, x - 1),
            (y, x - 1),
            (y + 1, x - 1),
            (y + 1, x),
            (y + 1, x + 1),
            (y, x + 1),
            (y - 1, x + 1),
            (y - 1, x)
        ]

        for cord in cords:
            if not output[cord] and edges[cord] > threshold_two:
                pixels.append(cord)

    return output
