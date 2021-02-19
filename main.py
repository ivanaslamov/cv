import cv2
import numpy as np
from scipy.optimize import minimize


image = cv2.imread("data/line.png")

if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
elif len(image.shape) == 4:
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
else:
    gray = image.clone()

height, width = gray.shape

start_point = np.array([200, 150, 200, 250])


def obj_func(p):
    value0 = 255 - gray[int(p[1]), int(p[0])]
    value1 = 255 - gray[int(p[3]), int(p[2])]
    dist = np.sqrt( (p[3] - p[1])**2 + (p[0] - p[2])**2 )
    return value0 + value1 + (100 - dist)*0.01


res = minimize(obj_func, start_point, options={'eps': 10})

x0 = int(res.x[0])
y0 = int(res.x[1])
x1 = int(res.x[2])
y1 = int(res.x[3])

image = cv2.circle(image, (x0, y0), 5, (255, 0, 0), 5)
image = cv2.circle(image, (x1, y1), 5, (255, 255, 0), 5)

cv2.imshow('frame', image)
cv2.waitKey(10000)

cv2.destroyAllWindows()
