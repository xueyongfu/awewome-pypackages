# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('ch04-图片/IMG_0000.jpg',cv2.IMREAD_COLOR)
# img = cv2.imread('ch04-图片/messi5.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('ch04-图片/messi5.jpg',cv2.IMREAD_UNCHANGED) #包括图像的 alpha 通道

# img = cv2.resize(img, (2400, 1800))

rows,cols,ch=img.shape
print('行/高:',rows,'列/宽:',cols,'通道:',ch)

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)#可以调整窗口大小
# cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)#自动调整
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)#保持图片比例

cv2.imshow('image', img)#窗口会自动调整为图像大小
# 在窗口上按任意键退出
cv2.waitKey(delay=0)#返回按键的 ASCII 码值
cv2.destroyAllWindows()

