import cv2
import os
#要提取视频的文件名，隐藏后缀
sourceFileName='2'
#在这里把后缀接上
vedio_path = os.path.join("vedio", sourceFileName+'.mp4')
times=0
#提取视频的频率，每25帧提取一个
frameFrequency=10
#输出图片到当前目录vedio文件夹下
outPutDirName='img/data4/'
if not os.path.exists(outPutDirName):
    #如果文件目录不存在则创建目录
    os.makedirs(outPutDirName) 
camera = cv2.VideoCapture(vedio_path)
while True:
    times+=1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times%frameFrequency==0:
        cv2.imwrite(outPutDirName + str(int(times/10))+'.jpg', image)
        print(outPutDirName + str(int(times/10))+'.jpg')
print('图片提取结束')
camera.release()