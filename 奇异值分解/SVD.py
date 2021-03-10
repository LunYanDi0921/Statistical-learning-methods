from PIL import Image
import numpy as np
import pandas as pd
import csv
from datetime import datetime
import time
import math
class SVD:
    def __init__(self,A,k):
        #待分解的矩阵A
        self.A=A
        self.k=k
        self.u,self.s1,self.vt,self.r=self.cal_value_and_vector()
    def cal_value_and_vector(self):
        #返回特征值与特征向量
        u, s, vt = np.linalg.svd(self.A)
        s1=np.array([[0]*self.A.shape[1]]*self.A.shape[0])
        k=0
        # print(s)
        # print(s.shape)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if i==j:
                   s1[i][j]=s[k]
                   k+=1
        return u,s1,vt,s.shape[0]

    def compress_picture(self):
        uk=self.u[:,0:self.k]
        s1k=self.s1[0:self.k,0:self.k]
        vk=self.vt.T[:,0:self.k]
        vkt=vk.T
        return np.dot(np.dot(uk,s1k),vkt)

def readData(): #返回一张照片的矩阵形式
    im = Image.open("J1.jpg")
    # 显示图片
    #im.show()
    im = im.convert("L")  # 灰色图像
    data = np.asarray(im)
    return data

def show_image(data):
     image = Image.fromarray(data)  # 将之前的矩阵转换为图片
     image=image.convert("RGB")
     image.show()  # 调用本地软件显示图片，win10是叫照片的工具

A=readData()
print(A)
svd=SVD(A,10)
print(svd.r)
data=svd.compress_picture()
show_image(data)
