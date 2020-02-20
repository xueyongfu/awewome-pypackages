# -*- coding: utf-8 -*-
'''
Otsu's 二值化

第一种方法 我们 设127 为全局 阈值。
第二种方法 我们直接使用 Otsu 二值化。
第三种方法 我 们 先使用一个 5x5 的 高斯核 去噪  然后再使用 Otsu 二值化。
看看噪音 去除对结果的影响有多大吧。
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ch15-图像阈值/noisy2.png', 0)

# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
# 5,5 为 斯核的大小 0 为标准差
blur = cv2.GaussianBlur(img, (5, 5), 0)
# 阀值一定为 0
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
# 使用了 pyplot 中画直方图的方法 plt.hist,
# 注意的是它的参数是一维数组
# 所以使用了 numpy ravel 方法 将多维数组 换成一维 也可以使用 flatten 方法
# ndarray.flat 1-D iterator over an array.
# ndarray.flatten 1-D array copy of the elements of an array in row-major order.

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()


#结果:先去噪,在二值化的效果最好