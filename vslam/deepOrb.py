import cv2
import numpy as np
from deeplab import DeeplabV3
from PIL import Image

class deepOrb:
    __name_classes = ["aeroplane", "bicycle", "bird", "boat", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "sheep", "train"]
    __orb = cv2.ORB_create()
    __deeplab = DeeplabV3()
    __padnum = 15

    @classmethod
    def deep_create(self): # 实例方法
        return deepOrb()
    
    def detectAndCompute(self, image):
        kps, des = self.__orb.detectAndCompute(image,None)
        # 过滤kp,des
        kps, des = self.__filter(kps, des, image)
        return kps, des
    
    def __filter(self, kps, des, image):
        padnum = self.__padnum
        padall = self.__padnum * 2 + 1
        matrix_shape = (padall, padall * 3)
        PIL_image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        r_image = self.__deeplab.detect_image(PIL_image, count=False, name_classes=self.__name_classes)
        cv2_img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
        
        # print("填充前：", cv2_img.shape)
        # cv2_img = np.pad(cv2_img,((padnum,padnum),(padnum,padnum),(0,0)),mode='constant', constant_values=0)
        # cv2.imshow("yuyi", cv2_img) #拼接显示为gray
        # cv2.waitKey(0)
        # print("切片前：", cv2_img.shape)
        kp_blue = [] #保留的特征点部分
        kp_red = []  #去除的特征点部分

        de_blue = []
        de_red = []

        # 扩大选区
        for i in range(0, len(kps)):
            kp = kps[i]
            de = des[i]
            pt =  kp.pt
            new_pt = (int(pt[0]), int(pt[1]))
            # print("切片操作：",new_pt[1] ,new_pt[1] + padall , new_pt[0] ,new_pt[0] + padall,0,2)
            point = cv2_img[new_pt[1]: new_pt[1] + padall , new_pt[0]:new_pt[0] + padall ,:]
            # print("切片后：",point.shape)

            point = point.reshape(matrix_shape)
            
            factor_bef = np.ones((1,matrix_shape[0]), dtype=int)
            factor_arf = np.ones((matrix_shape[1],1), dtype=int)
            
            if (factor_bef.dot(point).dot(factor_arf) == 0).all(): 
                kp_blue.append(kp)
                de_blue.append(de)
            else:
                kp_red.append(kp)
                de_red.append(de)
        return tuple(kp_blue), np.array(de_blue)
