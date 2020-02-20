"""
图像相减3.py:

3张图片

"""
import cv2

def diff(img, img1):  # returns just the difference of the two images
    # cv2.absdiff()获取差分图 就是将两幅图像作差
    return cv2.absdiff(img, img1)


def diff_remove_bg(img0, img, img1):  # removes the background but requires three images
    d1 = diff(img0, img)
    d2 = diff(img, img1)
    return cv2.bitwise_and(d1, d2)


# img1=cv2.imread('subtract1.jpg')
img1 = cv2.imread('ch10-图像上的算术运算/subtract1.jpg', 0)  # 灰度图
# img2=cv2.imread('subtract2.jpg')
img2 = cv2.imread('ch10-图像上的算术运算/subtract2.jpg', 0)

cv2.imshow('subtract1', img1)
cv2.imshow('subtract2', img2)

# 比较cv2.absdiff()和cv2.subtract()差异
# sub = cv2.subtract(img1, img2)
# dif = cv2.absdiff(img1, img2)
# cv2.imshow('sub', sub)
# cv2.imshow('dif', dif)

st = diff_remove_bg(img2, img1,img2)

cv2.imshow('after subtract', st)

cv2.waitKey(0)