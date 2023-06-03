import numpy as np
import cv2
from matplotlib import pyplot as plt
if __name__ == "__main__":

    orb = cv2.ORB_create()
    image1 = cv2.imread("img/data6/1341846314.325981.png")
    image2 = cv2.imread("img/data6/1341846314.357905.png")
    
    hmerge = np.hstack((image1, image2)) #水平拼接
    cv2.imshow("rgb", hmerge) #拼接显示为rgb
    cv2.waitKey(0)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #灰度处理图像
    kp1, des1 = orb.detectAndCompute(image1,None)#des是描述子

    
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(image2,None)

    hmerge = np.hstack((gray1, gray2)) #水平拼接
    cv2.imshow("gray", hmerge) #拼接显示为gray
    cv2.waitKey(0)
    for kp in kp1:
        print(kp.pt)

    img3 = cv2.drawKeypoints(image1,kp1,image1,color=(255,0,255))
    img4 = cv2.drawKeypoints(image2,kp2,image2,color=(255,0,255))

    cv2.imshow("img3",img3)
    cv2.imshow("img4",img4)
    cv2.waitKey(0)

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
    cv2.imshow("ORB", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()