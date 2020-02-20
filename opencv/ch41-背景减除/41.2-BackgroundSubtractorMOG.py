
"""
背景减除:我们需要从静止的背景中提取移动的前景。

这是一个以混合高斯模型为基础的前景/背景分割算法。
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)#笔记本摄像头

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# 可选参数 比如 进行建模场景的时间长度 高斯混合成分的数量-阈值等

while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, flipCode=1)  # 左右翻转

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(1) 
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
