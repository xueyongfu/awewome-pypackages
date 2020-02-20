
"""
函数 cv2.matchShape() 可以帮我们比较两个形状或轮廓的相似度。
如果返回值越小， 匹配越好。它是根据 Hu 矩来计算的。文档中对不同的方法有解释。
"""

import cv2
import numpy as np

img1 = cv2.imread('ch21-轮廓Contours/star.jpg', 0)
img2 = cv2.imread('ch21-轮廓Contours/star2.jpg', 0)

ret, thresh = cv2.threshold(img1, 127, 255, 0)
ret, thresh2 = cv2.threshold(img2, 127, 255, 0)

image,contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt1 = contours[0]
image,contours, hierarchy = cv2.findContours(thresh2, 2, 1)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
print(ret)

# Hu矩是归一化中心矩的线性组合
# 之所以这样做是为了能够获取代表图像的某个特征的矩函数
# 这些矩函数对某些变化如缩放 旋转 镜像映射（除了 h1）具有不变形。