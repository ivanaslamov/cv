import cv2
import numpy as np
from scipy.optimize import minimize
from scipy import ndimage

from app.utils import gradient_magnitude, distance, curvature

# image = cv2.imread("data/dark_circle.png")
#
# if len(image.shape) == 3:
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# elif len(image.shape) == 4:
#     gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
# else:
#     gray = image.clone()
#
# height, width = gray.shape
#
#
# # start points
# start_points = [
#     (250, 150),
#     (250, 200),
#     (250, 250),
#     (250, 300),
#     (250, 350),
# ]
#
# start_points_length = len(start_points)
#
# flatten_start_points = []
# for x, y in start_points:
#     flatten_start_points.append(x)
#     flatten_start_points.append(y)
#
# start_point = np.array(flatten_start_points)
#
# # gradient magnitude
# mag = 255 - gradient_magnitude(gray, kernel_size=32)*100
#
# color_mag = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
#
# def obj_func(p):
#     # draw
#     global color_mag
#
#     temp = color_mag.copy()
#
#     for i in range(0, len(p), 2):
#         x = int(p[i])
#         y = int(p[i + 1])
#         temp = cv2.circle(temp, (x, y), 5, (255, 0, 0), 5)
#
#     # display results
#     cv2.imshow('frame', temp)
#     cv2.waitKey(10)
#
#     # continuity
#     sum_distance = 0
#
#     for i in range(0, len(p)-2, 2):
#         sum_distance += distance(p[i], p[i+1], p[i+2], p[i+3])
#
#     mean_distance = 2 * sum_distance / len(p)
#
#     continuity_acc = (mean_distance - sum_distance)**2
#
#     # curvature
#     curvature_acc = 0
#
#     for i in range(2, len(p)-2, 2):
#         curvature_acc += curvature(p[i-2], p[i-1], p[i], p[i+1], p[i+2], p[i+3])
#
#     # magintude
#     mag_acc = 0
#
#     for i in range(0, len(p), 2):
#         x0 = int(p[i])
#         y0 = int(p[i+1])
#
#         value = mag[y0, x0]
#
#         print(value)
#
#         mag_acc += value
#
#     print(mag_acc)
#
#     # return value0 + value1 + value2 + abs(10 - dist0)*0.1 + abs(10 - dist1)*0.1
#     return mag_acc # continuity_acc + curvature_acc + mag_acc
#
# res = minimize(obj_func, start_point, options={'eps': 3})
# print(res.x)
#
# print(start_points)
#
# # print results
# # for x, y in start_points:
# #     color_mag = cv2.circle(color_mag, (x, y), 5, (255, 255, 0), 5)
#
# color_mag = color_mag.copy()
#
# for i in range(0, len(res.x), 2):
#     x = int(res.x[i])
#     y = int(res.x[i+1])
#     color_mag = cv2.circle(color_mag, (x, y), 5, (255, 0, 0), 5)
#
# # display results
# cv2.imshow('frame', color_mag)
# cv2.waitKey(10000)
#
# # clean results
# cv2.destroyAllWindows()

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import feature

# read image
img = cv2.imread("data/bird.jpg")

# pick initial points
height, width, channels = img.shape

points = []


def click_and_crop(event, x, y, _flags, _param):
    if event == 1:
        points.append((x, y))


cv2.namedWindow("img")
cv2.setMouseCallback("img", click_and_crop)

while (True):
    frame = img.copy()

    for point in points:
        frame = cv2.ellipse(frame, point, (2, 2), 5, 0, 360, 255, 1)

    cv2.imshow('img', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


edges = cv2.Canny(img, 100, 200)

edge_dist = ndimage.distance_transform_edt(~edges)



# s = np.linspace(0, 2*np.pi, 400)
# r = 200 + 180*np.sin(s)
# c = 320 + 270*np.cos(s)
# init = np.array([r, c]).T
#
# snake = active_contour(gaussian(img, 3),
#                        init, alpha=0.015, beta=10, gamma=0.001)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
# ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
#
# plt.show()

cv2.imshow('img', edge_dist)
cv2.waitKey(10000)

# clean results
cv2.destroyAllWindows()

# fig, ax = plt.subplots(figsize=(1, 1))
# ax.imshow(edge_dist, cmap=plt.cm.gray)
# plt.show()
