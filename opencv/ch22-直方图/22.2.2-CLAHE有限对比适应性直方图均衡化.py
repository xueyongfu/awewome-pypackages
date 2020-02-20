
"""
22.2.2-CLAHE有限对比适应性直方图均衡化.py:

自适应的直方图均衡化。这种情况下,整幅图像会分成很多小块,这些小块被称为tiles
在 OpenCV中tiles的大小是8x8
然后再对每一个小块分别进行直方图均衡化.
所以在每一个的区域中,直方图会 集中在某一个小的区域中
"""

import numpy as np
import cv2

img = cv2.imread('ch22-直方图/tsukuba_l.png', 0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

cv2.imshow('src', img)
cv2.imshow('clahe_2.jpg', cl1)
cv2.waitKey(0)

