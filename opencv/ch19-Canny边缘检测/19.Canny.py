'''
它是一个有很多步构成的算法
由于边缘检测很容易受到噪声影响 所以第一步是使用 5x5 的高斯滤波器去噪声
对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数的图像梯度Gx和Gy
梯度的方向一般总是与边界垂直。
梯度方向归为四类： 垂直 水平 和 两个对角线。
非极大值抑制
滞后阈值
现在确定那些边界才是真正的边界。这时我们设置两个阈值minVal和maxVal。
当图像的灰度梯度大于maxVal时为是真的边界
那些低于minVal的边界会被抛弃。
如果介于两者之间的 就看这个点是否与某个被确定为真正的边界点相连
如果是就认为它也是边界点 如果不是 就抛弃。

OpenCV 中的 Canny 边界检测
在OpenCV中只需要一个函数 cv2.Canny() 就可以完成以上几步。
我们看如何使用这个函数。
第一个参数是输入图像。
第二和第三个分别是 minVal 和 maxVal。
第三个参数用来计算图像梯度的Sobel卷积核的大小,默认值为3。
最后一个参数是 L2gradient 它可以用来 设定 求梯度大小的方程。
如果为True 就会使用我们上面提到的方程 否则 使用方程 Edge−Gradient (G) = |G2x| + |G2y| 代替，默认值为 False。
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ch19-Canny边缘检测/messi5.jpg',0)
edges = cv2.Canny(img, 100, 200)

cv2.imshow('Edges',edges)
cv2.waitKey(0)
