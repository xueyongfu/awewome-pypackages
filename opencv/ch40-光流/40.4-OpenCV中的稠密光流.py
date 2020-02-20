
"""
40.4-OpenCV中的稠密光流.py:

使用所有像素点的光流

"""

import cv2
import numpy as np

cap = cv2.VideoCapture("ch40-光流/slow.flv")
# cap = cv2.VideoCapture("../data/slow.flv")
ret, frame1 = cap.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', frame2)
    cv2.imshow('flow', bgr)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imshow('opticalfb.png', frame2)
        cv2.imshow('opticalhsv.png', bgr)
    prvs = next

cap.release()
cv2.destroyAllWindows()
