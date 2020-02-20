# -*- coding: utf-8 -*-
'''
仿射变换
在仿射变换中 原图中所有的平行线在结果图像中同样平行。
为了创建这个矩阵，我们需要从原图像中找到三个点以及他们在输出图像中的位置。
然后 cv2.getAffineTransform 会创建一个 2x3 的矩  最后 个矩 会 传给 函数 cv2.warpAffine。
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ch14-几何变换/drawing.png')
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('dst',dst)
cv2.waitKey(0)  
cv2.destroyAllWindows()

