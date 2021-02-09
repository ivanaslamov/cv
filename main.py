import cv2
import scipy as sp
import numpy as np
from scipy.optimize import line_search

from app.edges import canny

# from app.corners import shi, eigen

# read image
image = cv2.imread("data/checker_pattern.jpg")

# select starting point

















# find all non zero points

points = []


def obj_func(p):
    acc = 0
    for x, y in points:
        acc += p[0] * x ** 2 + p[1] * x * y + p[2] * y ** 2 + p[3] * x + p[4] * y + p[5]
    return acc


def obj_grad(_):
    acc_a = 0
    for x, _ in points:
        acc_a += x ** 2

    acc_b = 0
    for x, y in points:
        acc_b += x * y

    acc_c = 0
    for _, y in points:
        acc_c += y ** 2

    acc_d = 0
    for x, _ in points:
        acc_d += x

    acc_e = 0
    for _, y in points:
        acc_e += y

    acc_f = 0
    for _ in points:
        acc_f += 1

    return [acc_a, acc_b, acc_c, acc_d, acc_e, acc_f]


start_point = np.array([1.8, 1.7])
search_gradient = np.array([-1.0, -1.0])

results = line_search(obj_func, obj_grad, start_point, search_gradient)



a = - sqrt(2 * (A*E**2 + C * D**2 - B*D*E+(B**2-4*A*C)*F)*( (A+C)+sqrt( (A-C)**2+B**2 ) )) / (B**2 - 4 * A * C)
b = - sqrt(2 * (A*E**2 + C * D**2 - B*D*E+(B**2-4*A*C)*F)*( (A+C)-sqrt( (A-C)**2+B**2 ) )) / (B**2 - 4 * A * C)

x0 = (2 * C * D - B * E) / (B**2 - 4 * A * C)
y0 = (2 * A * E - B * D) / (B**2 - 4 * A * C)

theta = arctan( 1 / B * (C - A - sqrt((A - C)**2 + B**2)) )


color =
thickness =
image = cv2.ellipse(image, [x0,y0], [a,b], theta, 0, 360, color, thickness)


print(results)

corners = canny(image, 10, 5)

cv2.imshow('image', corners)
cv2.waitKey()

# cap = cv2.VideoCapture(0)
#
# while(True):
#     # Capture frame-by-frame
#     ret, image = cap.read()
#
#     # algorithm
#     edges = canny(image, 10, 5)
#
#     cv2.imshow('frame', edges)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()


cv2.destroyAllWindows()
