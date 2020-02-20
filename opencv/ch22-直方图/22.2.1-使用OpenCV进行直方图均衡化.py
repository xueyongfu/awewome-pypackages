"""
使用OpenCV进行直方图均衡化.py:
"""
import cv2
import numpy as np

img = cv2.imread('ch22-直方图/tsukuba_l.png', 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))  # stacking images side-by-side

cv2.imshow('src',img)
cv2.imshow('res.png', res)
cv2.waitKey(0)
'''
当直方图中的数据 中在某一个灰度值范围内时 直方图均 化很有用。 但是如果像素的变化很大 而且占据的灰度范围 常广时 
例如 既有很亮的 像素点又有很暗的像素点时
'''