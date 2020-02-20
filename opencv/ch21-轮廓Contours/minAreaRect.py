import cv2
import numpy as np

img = cv2.imread('ch21-轮廓Contours/lightning.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
print('len(contours)', len(contours))
cnt = contours[0]

#方向是物体定向的角度
(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)  #根据所有的表示轮廓的点拟合出一个椭圆,使这些点都在椭圆上
print((x, y), (MA, ma), angle)

rect = cv2.minAreaRect(cnt)  #生成最小外接矩形
box = cv2.boxPoints(rect)  #Finds the four vertices of a rotated rect, Useful to draw the rotated rectangle.
box = np.int0(box)
img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

cv2.imshow('fd', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
