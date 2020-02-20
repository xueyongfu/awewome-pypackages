
"""
orb.py:
SIFT 和 SURF 算法是有专利保护的，如果你要使用它们，就可能要
花钱。但是 ORB 不需要！！！
ORB 基本是 FAST 关键点检测和 BRIEF 关键点描述器的结合体，并通
过很多修改增强了性能。首先它使用 FAST 找到关键点，然后再使用 Harris
角点检测对这些关键点进行排序找到其中的前 N 个点。它也使用金字塔从而产
生尺度不变性特征。但是有一个问题，FAST 算法步计算方向。那旋转不变性
怎样解决呢？作者进行了如下修改。
它使用灰度矩的算法计算出角点的方向。以角点到角点所在（小块）区域
质心的方向为向量的方向。为了进一步提高旋转不变性，要计算以角点为中心
半径为 r 的圆形区域的矩，再根据矩计算除方向。
对于描述符，ORB 使用的是 BRIEF 描述符。但是我们已经知道 BRIEF
对与旋转是不稳定的。所以我们在生成特征前，要把关键点领域的这个 patch
的坐标轴旋转到关键点的方向。
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('ch36-ORB/blox.jpg', 0)

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img, None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

plt.imshow(img2), plt.show()
