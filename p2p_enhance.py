import cv2
import numpy as np
import glob
from vslam.deepOrb import deepOrb as deepOrbClass



"""
基于特征点匹配算法的2d-2d位姿估计
"""
if __name__ == "__main__":
    prediction_ground = []
    outputText = "./output/enhance_prediction_ground.txt"

    i = 0
    p00 = []
    p11 = []

    p_world = np.zeros((3,1))

    num = 0

    image_pre = cv2.imread("img/data6/1341846314.325981.png")
    image_new = cv2.imread("img/data6/1341846314.357905.png")

    orb = deepOrbClass.deep_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    principal_point = (320.1, 247.6)  #光心 TUM dataset标定值
    focal_length = 535.4          # 焦距TUM dataset标定值
    K = np.array((535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1)).reshape((3,3)) # 相机内参

    for imagePath in glob.glob("img/data6" + "/*.png"):
        timestamp = imagePath[10:-4]
        num += 1
        # 加载图像，转为灰度图

        if i < 1:
            image_pre = cv2.imread(imagePath)
            i += 1
            continue
        image_new = cv2.imread(imagePath)

        # hmerge = np.hstack((image_pre, image_new)) #水平拼接
        # cv2.imshow(imagePath, hmerge) #拼接显示为gray
        # cv2.waitKey(0)
        
        kp1, des1 = orb.detectAndCompute(image_pre)
        kp2, des2 = orb.detectAndCompute(image_new)
        # BFMatcher解决匹配
        matches11 = bf.knnMatch(des1,des2, k=2)
        # kp1, kp2, matches, matchesMask = RANSAC(image_pre, image_new, kp1, kp2, matches11)

        good = []
        for m,n in matches11:
            if m.distance < 0.75*n.distance:
                good.append([m])

        for j in range(len(good)):
            p00.append(list(kp1[good[j][0].queryIdx].pt))
            p11.append(list(kp2[good[j][0].trainIdx].pt))
        
            
        p000 = np.array(p00)
        p111 = np.array(p11)
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(p000, p111, cameraMatrix=K, method=cv2.RANSAC)
        # print(f'E:{E}')
        # 估计两帧之间的运动R,t
        _, R, t, _ = cv2.recoverPose(E, p000, p111, cameraMatrix=K)

        # 计算四元数
        # Convert the rotation matrix to a quaternion representation
        tr = np.trace(R)
        qw = np.sqrt(1 + tr) / 2
        qx = (R[2,1] - R[1,2]) / (4 * qw)
        qy = (R[0,2] - R[2,0]) / (4 * qw)
        qz = (R[1,0] - R[0,1]) / (4 * qw)
        q = np.array([qx, qy, qz, qw])
        q = q / np.linalg.norm(q)
        print(f'Q:{q}')

        p_world = np.dot(R, p_world) + t
        p_world2 = p_world*6/1662
        image_pre = image_new.copy()
        
        p00.clear()
        p11.clear()
        string = f'{timestamp} {"%.6f" % p_world[0]} {"%.6f" % p_world[1]} {"%.6f" % p_world[2]} {"%.6f" % q[0]} {"%.6f" % q[1]} {"%.6f" % q[2]} {"%.6f" % q[3]}'
        print(string)
        prediction_ground.append(string)
    sep = '\n'
    with open(outputText,"w") as f:
        f.write(sep.join(prediction_ground))


