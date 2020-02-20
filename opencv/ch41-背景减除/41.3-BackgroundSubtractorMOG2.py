"""
这个也是以高斯混合模型为基础的背景/前景分割算法。

这个算法的一个特点是它为每一个像素选择一个合适数目的高斯分布。（上一个方法中我们使用是 K 高斯分
布）。这样就会对由于亮度等发生变化引起的场景变化产生更好的适应。
"""

import numpy as np
import cv2

# cap = cv2.VideoCapture('../data/vtest.avi')
cap = cv2.VideoCapture(0)#笔记本摄像头

fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, flipCode=1)  # 左右翻转

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) #& 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
