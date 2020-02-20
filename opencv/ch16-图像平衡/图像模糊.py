"""
是由一个归一化卷积框完成的。用卷积框覆盖区域所有像素的平均值来代替中心元素。可以使用函数 cv2.blur() 和 cv2.boxFilter() 来完成个任务。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ch16-图像平衡/opencv_logo.png')
#kernel是均值的,功能相当于均值滤波
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)
# blur = cv2.blur(img, (5, 5))



'''
现在把卷积核换成高斯核,简单来说,方框不变 将原来每个方框的值是相等的 现在的值是符合高斯分布的,方框中心的值最大,其余方框根据 
离中心元素的距离递减,构成一个高斯小山包。原来的求平均数现在变成求加权平均数
'''
# 0 Gaussian kernel standard deviation in X direction
blur = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊




'''
 名思义就是用与卷积框对应像素的中值来替代中心像素的值。这个滤波器经常用来去椒盐噪声。
'''
median = cv2.medianBlur(img, 5)  # 中值模糊




'''
函数 cv2.bilateralFilter() 能在保持边界清晰的情况下有效的去噪,但是这种操作与其他滤波器相比会比较慢。
我们已经高斯滤波器是求与中心点邻近区域像素的高斯加权平均值。这种高斯滤波器只考虑像素之间的空间关系,而不会考虑像素值之间的关系或者像素的相似度
所以这种方法不会考虑一个像素是否位于边界。因此边界也会被模糊掉,而这正不是我们想要。

双边滤波在同时使用空间高斯权重和灰度值相似性。空间高斯函数确保只有邻近区域的像素对中心点有影响,灰度值相似性高斯函数确保只有与中心像素灰度值
相近的才会被用来做模糊运算。所以这种方法会确保边界不会被模糊掉,因为边界处的灰度值变化比较大。
'''

# cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# d – Diameter of each pixel neighborhood that is used during filtering. # If it is non-positive, it is computed from sigmaSpace
# 9  域直径 两个 75 分别是空  斯函数标准差 灰度值相似性 斯函数标准差
blur = cv2.bilateralFilter(img, 9, 75, 75)

