# coding=utf-8

import cv2
import numpy as np

if __name__ == "__main__":
    # 读取图像
    image1 = cv2.imread("img/data5/000168.png")
    image2 = cv2.imread("img/data5/000169.png")
    #灰度处理图像
    cv2.imshow("test",image1)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None) #des是描述子
    kp2, des2 = sift.detectAndCompute(image2,None) #des是描述子
    hmerge = np.hstack((gray1, gray2)) #水平拼接
    cv2.imshow("gray", hmerge) #拼接显示为gray
    cv2.waitKey(0)
     #画出特征点，并显示为红色圆圈
    img3 = cv2.drawKeypoints(image1,kp1,image1,color=(255,0,255))
    img4 = cv2.drawKeypoints(image2,kp2,image2,color=(255,0,255))
    hmerge = np.hstack((img3, img4)) #水平拼接
    cv2.imshow("point", hmerge) #拼接显示为gray
    cv2.waitKey(0)

    # BFMatcher解决匹配

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # 调整ratio
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img5 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good,None,flags=2)
    cv2.imshow("BFmatch", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()