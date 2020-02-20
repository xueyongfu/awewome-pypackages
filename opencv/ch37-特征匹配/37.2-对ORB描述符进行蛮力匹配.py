"""
37.2-对ORB描述符进行蛮力匹配.py:
蛮力匹配器是很简单的。首先在第一幅图像中选取一个关键点然后依次与
第二幅图像的每个关键点进行（描述符）距离测试，最后返回距离最近的关键
点。

匹配器对象是什么
matches = bf.match(des1, des2) 返回值是一个 DMatch对象列表
DMatch 对 具有下列属性
• DMatch.distance - 描述符之间的距离。越小越好。
• DMatch.trainIdx - 目标图像中描述符的索引。
• DMatch.queryIdx - 查询图像中描述符的索引。
• DMatch.imgIdx - 目标图像的索引。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('ch37-特征匹配/box.png', 0)  # queryImage
img2 = cv2.imread('ch37-特征匹配/box_in_scene.png', 0)  # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)  # 前10个匹配
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,flags=2)  # 前10个匹配

plt.imshow(img3), plt.show()
