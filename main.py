import cv2

from app.edges import canny
# from app.corners import shi, eigen


image = cv2.imread("data/beams.jpg")
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
