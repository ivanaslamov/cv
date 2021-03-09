import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the left and right images
# in gray scale
from app.matching import match

imgLeft = cv2.imread('data/image_l.jpg', 0)
imgRight = cv2.imread('data/image_r.jpg', 0)

match(imgLeft, imgRight, preview=True)
