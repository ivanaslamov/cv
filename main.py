import cv2
import scipy as sp
import numpy as np
from scipy.optimize import minimize, line_search

from app.edges import canny
from app.fitting import ellipse

# from app.corners import shi, eigen

# read image
image = cv2.imread("data/elipse.png")

kernel = ellipse(image)

cv2.imshow('frame', kernel)
cv2.waitKey(10000)

cv2.destroyAllWindows()
