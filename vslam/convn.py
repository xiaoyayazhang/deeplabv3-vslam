import numpy as np

def Convn(img, kernel, n, stride,p):
	#img：输入图片；kernel：卷积核值；n：卷积核大小为n*n；stride:步长。
	#return：feature map
    h, w = img.shape
    img = np.pad(img,((1,1),(1,1)),'constant',constant_values=0)
    res_h = ((h+2*p-n)//stride)+1 #卷积边长计算公式：((n+2*p-k)/stride)+1
    res_w = ((w+2*p -n)//stride)+1
    res = np.zeros([res_h, res_w])
    # print(res)
    for i in range(res_h):
        for j in range(res_w):
            temp = img[i*stride:i*stride+n , j*stride:j*stride+n]
            print((i*stride,i*stride+n,j*stride,j*stride+n)) #打印检查卷积核每次卷积的位置对否
            temp = np.multiply(kernel, temp)
            res[i][j] = temp.sum()
    return res
