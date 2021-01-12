import cv2

from app.edges import eigen


image = cv2.imread("data/beams.jpg")
corners = eigen(image)

cv2.imshow('image', corners)
cv2.waitKey()

cv2.destroyAllWindows()
