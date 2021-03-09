import cv2

from app.stereo import epipolar


imgLeft = cv2.imread('data/image_l.jpg', 0)
imgRight = cv2.imread('data/image_r.jpg', 0)

epipolar(imgLeft, imgRight)
