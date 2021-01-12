import cv2

from app.edges import eigen
from app.utils import gaussian

image = cv2.imread("data/beams.jpg")

window_size = 8
ones_helper = gaussian(window_size)

image = eigen(image)

cv2.imshow('image', image)
cv2.waitKey()

cv2.destroyAllWindows()
