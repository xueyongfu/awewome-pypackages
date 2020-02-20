
"""
函数 cv2.warpAffine() 的第三个参数的是 出图像的大小 ，它的格式 应 是图像的(宽,高) 。
图像的宽对应的是列数, 高对应的是行数。
"""

import cv2
import numpy as np

img = cv2.imread('ch14-几何变换/messi5.jpg', 1)
rows, cols, _ = img.shape

#x方向移动100像素点, y方向移动50像素点
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
