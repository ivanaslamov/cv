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
        frame = cv2.ellipse(frame, (point0_x, point0_y), (int(radius), int(radius)), 5, 0, 360, 255, -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


b = 0
c = 1
d = - 2*point0_x
e = - 2*point0_y
f = point0_y**2 + point0_x**2 - radius**2

# b, c, d, e, f = [0.00000e+00, 1.00000e+00, -5.96000e+02, -5.64000e+02, 1.51236e+05]
# b, c, d, e, f = [-1.92676280e+00,  1.35912594e+00, -5.96006178e+02, -5.64006227e+02, 1.51236000e+05]
start_point = np.array([b, c, d, e, f])


def obj_func(p):
    # print(p)

    acc = 0
    for x, y in points:
        # acc += (x**2 + p[0] * x * y + p[1] * y**2 + p[2] * x + p[3] * y + p[4])**2
        acc += (x ** 2 + p[0] * x * y + p[1] * y ** 2 + p[2] * x + p[3] * y + p[4]) ** 2

    # print(acc)
    return acc

def constraints(p):
    return [4 * p[1] - p[0]**2, p[2]**2/4 + p[3]**2 / (4 * p[2]) - p[4]]


res = minimize(obj_func, start_point, constraints={"fun": constraints, "type": "ineq"})


print(res)

b, c, d, e, f = res.x


# b, c, d, e, f = [-8.37005424e+09, -2.36930466e+11,  5.74366974e+09,  5.78838533e+09, 1.78155234e+07]

# # a, b, c, d, e, f = [ 9.36154278e-05,  0.00000000e+00,  9.36154278e-05, -5.03651002e-02, -5.59820258e-02,  1.51434188e+01]

kernel = np.zeros((height, width), np.float32)

maximum = 0

for y in range(0, height):
    for x in range(0, width):
        value = x**2 + b * x * y + c * y**2 + d * x + e * y + f
        if (value > maximum):
            maximum = value
        kernel[(y, x)] = np.abs(value)

kernel /= 1000

cv2.imshow('frame', kernel)
cv2.waitKey(10000)

#
# def click_and_crop(event, x, y, flags, param):
#     global point0_x, point0_y, radius
#
#     print(kernel[(y, x)])
#
#     # if flags:
#     #     radius = np.sqrt((point0_x - x) ** 2 + (point0_y - y) ** 2)
#     # elif radius is None:
#     #     point0_x = x
#     #     point0_y = y
#
# # select starting point
# cv2.namedWindow("frame")
# cv2.setMouseCallback("frame", click_and_crop)
#
# while(True):
#     cv2.imshow('frame', kernel)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break


# # find all non zero points
#
# points = []
#
#
# def obj_func(p):
#     acc = 0
#     for x, y in points:
#         acc += p[0] * x ** 2 + p[1] * x * y + p[2] * y ** 2 + p[3] * x + p[4] * y + p[5]
#     return acc
#
#
# def obj_grad(_):
#     acc_a = 0
#     for x, _ in points:
#         acc_a += x ** 2
#
#     acc_b = 0
#     for x, y in points:
#         acc_b += x * y
#
#     acc_c = 0
#     for _, y in points:
#         acc_c += y ** 2
#
#     acc_d = 0
#     for x, _ in points:
#         acc_d += x
#
#     acc_e = 0
#     for _, y in points:
#         acc_e += y
#
#     acc_f = 0
#     for _ in points:
#         acc_f += 1
#
#     return [acc_a, acc_b, acc_c, acc_d, acc_e, acc_f]
#
#
# start_point = np.array([1.8, 1.7])
# search_gradient = np.array([-1.0, -1.0])
#
# results = line_search(obj_func, obj_grad, start_point, search_gradient)
#
#
#
# a = - sqrt(2 * (A*E**2 + C * D**2 - B*D*E+(B**2-4*A*C)*F)*( (A+C)+sqrt( (A-C)**2+B**2 ) )) / (B**2 - 4 * A * C)
# b = - sqrt(2 * (A*E**2 + C * D**2 - B*D*E+(B**2-4*A*C)*F)*( (A+C)-sqrt( (A-C)**2+B**2 ) )) / (B**2 - 4 * A * C)
#
# x0 = (2 * C * D - B * E) / (B**2 - 4 * A * C)
# y0 = (2 * A * E - B * D) / (B**2 - 4 * A * C)
#
# theta = arctan( 1 / B * (C - A - sqrt((A - C)**2 + B**2)) )
#
#
# color =
# thickness =
# image = cv2.ellipse(image, [x0,y0], [a,b], theta, 0, 360, color, thickness)
#
#
# print(results)
#
# corners = canny(image, 10, 5)
#
# cv2.imshow('image', corners)
# cv2.waitKey()
#
# # cap = cv2.VideoCapture(0)
# #
# # while(True):
# #     # Capture frame-by-frame
# #     ret, image = cap.read()
# #
# #     # algorithm
# #     edges = canny(image, 10, 5)
# #
# #     cv2.imshow('frame', edges)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # cap.release()

cv2.destroyAllWindows()
