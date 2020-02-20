'''
原理
梯度简单来说就是求导。
OpenCV 提供了三种不同的梯度滤波器 或者说是高通滤波器：Sobel，  Scharr 和 Laplacian。我们会意义介绍他们。
Sobel Scharr 其实就是求一阶或二阶导数。
Scharr 是对 Sobel (使用小的卷积核求解 梯度角度时 )的优化。
Laplacian 是求二阶导数。


Sobel 算子是高斯平滑与微分操作的结合体 所以它的抗噪声能力很好。
 你可以设定求导的方向 xorder 或 yorder 。
 可以设定使用的卷积核的大 小 ksize 。
 如果 ksize=-1 会使用 3x3 的 Scharr 滤波器
 它的的效果比3x3的Sobel滤波器好 而且速度相同 所以在使用3x3滤波器时应尽量使用 Scharr滤波器 。

Laplacian 算子
拉普拉斯算子可以使用二阶导数的形式定义 ，可假设其离散实现类似于二阶Sobel导数
事实上OpenCV 在算拉普拉斯算子时直接用Sobel算子
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('ch18-图像梯度/sudoku.jpg', 0)



laplacian = cv2.Laplacian(img, cv2.CV_64F)


# 参数 1,0 为只在 x 方向求一 导数 最大可以求 2 导数。
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)


# 参数 0,1 为只在 y 方向求一 导数 最大可以求 2 导数。
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)


sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)


plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5), plt.imshow(sobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.show()
