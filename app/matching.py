# the following code is taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

import cv2
from matplotlib import pyplot as plt


def match(imgLeft, imgRight, preview=False):
    # Detect the SIFT key points and
    # compute the descriptors for the
    # two images
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgLeft, None)
    kp2, des2 = sift.detectAndCompute(imgRight, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    if preview:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(imgLeft, kp1, imgRight, kp2, good, None, flags=2)

        plt.imshow(img3), plt.show()

    return (kp1, kp2, good)
