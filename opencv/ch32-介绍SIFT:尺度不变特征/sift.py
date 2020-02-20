"""
引入:在前面两节我们学习了一些角点检测技术，比如 Harris 等。它们具有旋
转不变特性，即使图片发生了旋转，我们也能找到同样的角点。很明显即使图
像发生旋转之后角点还是角点。那如果我们对图像进行缩放呢？角点可能就不
再是角点了。以下图为例，在一副小图中使用一个小的窗口可以检测到一个角
点，但是如果图像被放大，再使用同样的窗口就检测不到角点了。

尺度不变特征变换

关键点 极值点 定位
"""

import cv2
import numpy as np

img = cv2.imread('ch32-介绍SIFT/home.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, img)

# 计算关键点描述符
# 使用函数 sift.compute() 来 计算 些关键点的描述符。例如
# kp, des = sift.compute(gray, kp)
kp, des = sift.detectAndCompute(gray,None)

cv2.imshow('sift_keypoints.jpg', img)
cv2.waitKey(0)
