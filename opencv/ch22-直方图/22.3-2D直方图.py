
"""
22.3-2D直方图.py:
一维是因为 我们只考了图像的一个特征：灰度值。
但是在2D直方图中我们就考虑两个图像特征。

对于彩色图像的直方图通常情况下我们考每个的颜色Hue 和 饱和度 Saturation 。根据两个特征绘制2D 直方图。

使用函数cv2.calcHist()来算直方图既简单又方便。如果绘制颜色直方图的话， 我们先将图像的颜色空间从BGR换到HSV计算2D直方图
记住计算一维直方图从BGR换到 HSV 。
函数的参数做如下修改
• channels=[0 1] 因为我们同时处理 H 和 S 两个 。
• bins=[180 256]H为 180 S为 256。
• range=[0 180 0 256]H 的取值范围在 0 到 180 S 的取值范围 在 0 到 256。

"""

import cv2
import numpy as np

img = cv2.imread('ch22-直方图/home.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Numpy 同样提供了绘制 2D 直方图的函数 np.histogram2d()。
# 绘制 1D 直方图时我们使用的是 np.histogram() 。

h, s, v = cv2.split(hsv)

hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])

pass
