'''
两个基本的形态学操作是腐蚀和膨胀。他们的变体构成了开运算，闭运算，梯度等。

注意:腐蚀和膨胀都是针对白色区域而言的
'''

import cv2
import numpy as np

img = cv2.imread('ch17-形态学转换/j.png', 0)

#您可以将内核看作是一个小矩阵，我们在图像上滑动以进行（卷积）操作，例如模糊，锐化，边缘检测或其他图像处理操作。
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('j.png', img)
cv2.imshow('erode', erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()
