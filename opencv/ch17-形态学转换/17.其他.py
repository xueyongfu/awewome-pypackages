import cv2
import numpy as np

img = cv2.imread('ch17-形态学转换/j.png', 0)
cv2.imshow('src', img)

#您可以将内核看作是一个小矩阵，我们在图像上滑动以进行（卷积）操作，例如模糊，锐化，边缘检测或其他图像处理操作。
kernel = np.ones((5, 5), np.uint8)


# 开运算：先腐蚀再膨胀就叫做开运算 它用来去噪声。
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)


# 闭运算 先膨胀再腐 。它经常 用来填充前景物体中的小洞 或者前景物体上的小黑点。
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)


# 形态学梯度 其实就是一幅图像膨胀与腐蚀的差别。结果看上去就像前景物体的轮廓
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)


# 礼帽  原始图像与开运算之后得到的图像的差。
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)


# 黑帽  进行闭运算之后得到的图像与原始图像的差
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)


cv2.waitKey(0)
cv2.destroyAllWindows()

