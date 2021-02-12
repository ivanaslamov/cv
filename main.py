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


a = 1
b = 0
c = 1
d = - 2*point0_x
e = - 2*point0_y
f = point0_y**2 + point0_x**2 - radius**2

start_point = np.array([a, b, c, d, e, f])


def equation(p, x, y):
    return p[0] * x**2 + p[1] * x * y + p[2] * y**2 + p[3] * x + p[4] * y + p[5]


def obj_func(p):
    acc = 0

    for x, y in points:
        acc += (equation(p, x, y))**2

    return acc


res = minimize(obj_func, start_point)


print(res)


kernel = np.zeros((height, width), np.float32)

maximum = 0

for y in range(0, height):
    for x in range(0, width):
        value = equation(res.x, x, y)
        if (value > maximum):
            maximum = value
        kernel[(y, x)] = value # np.abs(value)

cv2.imshow('frame', kernel)
cv2.waitKey(10000)

cv2.destroyAllWindows()
