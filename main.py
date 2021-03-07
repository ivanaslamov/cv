import cv2
import numpy as np
from scipy import ndimage

from app.fitting import snake


# read image
img = cv2.imread("data/dark_square.png")

# pick initial points
height, width, channels = img.shape

points = []


def click_and_crop(event, x, y, _flags, _param):
    if event == 1:
        points.append((y, x))


cv2.namedWindow("img")
cv2.setMouseCallback("img", click_and_crop)

while (True):
    frame = img.copy()

    for point in points:
        frame = cv2.ellipse(frame, (point[1], point[0]), (2, 2), 5, 0, 360, 255, 1)

    cv2.imshow('img', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

points = np.array(points)

edges = cv2.Canny(img, 100, 200)

edge_dist = ndimage.distance_transform_edt(~edges)

snake_pts = snake(points, edge_dist, alpha=1, beta=0.05, nits=1000, img=img)

# display results
frame = img.copy()

for point in snake_pts:
    frame = cv2.circle(frame, (int(point[1]), int(point[0])), 5, (255,0,0), 2)

cv2.imshow('img', frame)
cv2.waitKey(10000)

# clean results
cv2.destroyAllWindows()
