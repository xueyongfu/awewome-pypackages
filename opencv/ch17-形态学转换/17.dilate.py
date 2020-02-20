import cv2
import numpy as np

img = cv2.imread('ch17-形态学转换/j.png', 0)

kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('j.png', img)
cv2.imshow('dilation', dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
