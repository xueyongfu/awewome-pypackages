
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ch22-直方图/home.jpg', 0)
# img.ravel() 将图像转成一维数组
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
