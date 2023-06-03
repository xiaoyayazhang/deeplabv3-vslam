import cv2
import numpy as np
from deeplab import DeeplabV3
from PIL import Image

if __name__ == "__main__":
    imgpath = "img/data3/15.jpg"
    image1 = cv2.imread(imgpath)

    ################################
    # 1.对图片进行特征提取          #
    ################################
    orb = cv2.ORB_create()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    kps, des = orb.detectAndCompute(image1,None)

    #############################################
    # 2.做语义分析，过滤出移动物体所含有的像素    #
    #############################################
    deeplab = DeeplabV3()
    name_classes = ["aeroplane", "bicycle", "bird", "boat", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "sheep", "train"]
    
    image = Image.open(imgpath)
    #生成的图,扣除背景图
    r_image = deeplab.detect_image(image, count=False, name_classes=name_classes)
    r_image.show()
    cv2_img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
    
    ################################
    # 3.过滤掉移动物体的特征点      #
    ################################
    kp_blue = [] #保留的特征点部分
    kp_red = []  #去除的特征点部分

    for kp in kps:
        pt =  kp.pt
        new_pt = (int(pt[0]), int(pt[1]))
        point_R = cv2_img[new_pt[1]][new_pt[0]][0]
        point_G = cv2_img[new_pt[1]][new_pt[0]][0]
        point_B = cv2_img[new_pt[1]][new_pt[0]][0]

        point_pixel = np.array((point_R, point_G, point_B))
        factor = np.ones((3,1), dtype=int)
        print("point_pixel", point_pixel)
        
        if point_pixel.dot(factor) == 0: 
            print("0-dot" ,point_pixel.dot(factor))
            # img_kp = cv2.circle(img_kp, new_pt, 1, (0, 6, 255), thickness=-1)
            kp_blue.append(kp)
        else:
            print("1-dot" ,point_pixel.dot(factor))
            kp_red.append(kp)
            # img_kp = cv2.circle(img_kp, new_pt, 1, (4, 98, 228), thickness=-1)

    ################################
    # 4.绘画出特征点                #
    ################################
    img_kp = cv2.drawKeypoints(image1,tuple(kp_blue),1,color=(255, 0, 0))
    img_kp = cv2.drawKeypoints(img_kp,tuple(kp_red),1,color=(0, 6, 255))

    cv2.imshow("img_kp", img_kp)
    cv2.waitKey(0)
    image_kp = Image.fromarray(np.uint8(img_kp))
    image_mix = Image.blend(image_kp, r_image, 0.4)
    image_mix.show()