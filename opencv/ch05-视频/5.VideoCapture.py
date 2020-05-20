
'''
注意 当你的程序报错时 你先检查的是你的摄像头是否能够在其他程 序中正常工作 比如 linux 下的 Cheese。
'''

import numpy as np
import cv2

cap = cv2.VideoCapture(1)

cap.set(3,1600)
cap.set(4,720)

# cv2.namedWindow('',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('', 720, 680)

while cap.isOpened():  
    # Capture frame-by-frame
    ret, frame = cap.read()  # ret返回一个布尔值 True/False
    print(('frame shape:',frame.shape))

    cv2.imshow("frame", frame)

    key = cv2.waitKey(delay=10)
    if key == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
