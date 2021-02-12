import cv2
import scipy as sp
import numpy as np
from scipy.optimize import minimize, line_search

from app.edges import canny

# from app.corners import shi, eigen

# read image
image = cv2.imread("data/elipse.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

height, width = gray.shape

points = []

for y in range(0, height - 1):
    for x in range(0, width - 1):
        if gray[y, x] > 100:
            points.append((x, y))

point0_x = None
point0_y = None
radius = None

def click_and_crop(event, x, y, flags, param):
    global point0_x, point0_y, radius

    if flags:
        radius = np.sqrt((point0_x - x) ** 2 + (point0_y - y) ** 2)
    elif radius is None:
        point0_x = x
        point0_y = y

# select starting point
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_and_crop)

while(True):
    frame = gray.copy()

    if radius:
        frame = cv2.ellipse(frame, (point0_x, point0_y), (int(radius), int(radius)), 5, 0, 360, 255, 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


start_point = np.array([point0_x, point0_y, radius])


def obj_func(p):
    acc = 0

    for x, y in points:
        acc += ( (x - p[0])**2 + (y - p[1])**2 - p[2]**2 )**2

    return acc


res = minimize(obj_func, start_point)


print(res)


a, b, c = res.x

kernel = np.zeros((height, width), np.float32)

maximum = 0

for y in range(0, height):
    for x in range(0, width):
        value = (x - a)**2 + (y - b)**2 - c**2
        if (value > maximum):
            maximum = value
        kernel[(y, x)] = value # np.abs(value)

cv2.imshow('frame', kernel)
cv2.waitKey(10000)

cv2.destroyAllWindows()
