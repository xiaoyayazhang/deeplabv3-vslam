import cv2
import numpy as np

# 设定棋盘格参数
pattern_size = (8, 6)
square_size = 0.02

# 读取棋盘格图片
img = cv2.imread("./img/tum_fr3_checkerboard/1341834870.608741.png")

# 寻找棋盘格角点
found, corners = cv2.findChessboardCorners(img, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

# 检查棋盘格角点是否成功寻找到
if not found:
    print("未能在棋盘格图片中找到角点")
    exit()

print(img.shape)
# 计算相机内参
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera([corners], (480, 640), pattern_size, square_size)

print(f'K:{K}')
print(f'dist:{dist}')
