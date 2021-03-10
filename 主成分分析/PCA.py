import pandas as pd
import csv
from datetime import datetime
import numpy as np
import time
import math
from sklearn.datasets import load_breast_cancer
from collections import defaultdict
from pandas import read_csv
from sklearn import preprocessing
from sklearn import datasets
import sys
class PCA:

    def __init__(self,Train,eta):
        # Train m * n 矩阵 其中 m 是有几个随机变量 n是有多少个样本 每个随机变量对应n个取值
        #对Train进行规范化处理
        # print(Train)
        self.Train=self.standardize_data(Train)
        #方差贡献率阈值
        self.eta=eta

    def standardize_data(self,Train):
        # 规范化数据
        # 求每一个随机变量的取值的均值
        avg=Train.sum(axis=1)/Train.shape[1]
        # 将avg变成 m*1 矩阵
        avg.resize([Train.shape[0], 1])
        print(avg)
        # 将每个随机变量的取值 减去均值 data 是随机变量均值矩阵
        data=Train-avg
        print(data)
        # var每个随机变量的方差矩阵
        #初始化var
        var=np.array([[0.0]]*Train.shape[0])
        #求解每个随机变量的标准差-方差开平方
        for i in range(Train.shape[0]):
            var[i][0]=math.sqrt(np.dot(data[i],data[i].T)/Train.shape[1])
        print(var)
        #得到规范化的数据
        data=data/var
        print(data)
        #返回数据
        return data

    def cal_R(self):
        # 计算样本相关矩阵R 当进行过标准化之后样本的相关矩阵就是协方差矩阵
        #print(self.Train)
        R=np.dot(self.Train,self.Train.T)
        print(R)
        return R/(self.Train.shape[1]-1)

    def cal_value_vector(self,R):
        # 求矩阵R的特征值和特征向量
        val, vector= np.linalg.eig(R)
        print(val)
        print(vector)
        return val,vector

    def choose_k(self,val):
       # 确定主成分的个数
       #计算方差之和
       var_sum=val.sum()
       sum=0
       k=0
       eta=sum/var_sum
       while eta<self.eta:
           sum +=val[k]
           k+=1
           eta = sum / var_sum
       print(k)
       return k

    def cal_k_vector(self,vector,k):
        #求前k个特征值的特征向量
        return vector[:,0:k]

    def cal_y(self,A):
        #计算k个主成分
        print(A)
        A_T=A.T
        print(A_T)
        y=np.array([[0.0]*A.shape[0]]*A.shape[1])
        for k in range(A.shape[1]):
            data=np.array(A_T[k])
            data.resize([1,A.shape[0]])
            print(data)
            print(self.Train.T)
            yk=data*self.Train.T
            print(yk)
            yk=yk.sum(axis=0)
            print(yk)
            y[k]=yk
        y_T=y.T
        return y_T

    def fit(self):
        R=self.cal_R()
        val, vector=self.cal_value_vector(R)
        k=self.choose_k(val)
        k_vector=self.cal_k_vector(vector,k)
        y=self.cal_y(k_vector)
        print(y)

iris = datasets.load_iris()
data=iris.data
Train=data.T
pca=PCA(Train,0.75)
pca.fit()
# Train=np.array([[1,2,3,2],
#                 [4,5,6,4],
#                 [2,4,1,10]])
# print(Train)

# print(Train)
# data=np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])
# data1=data.sum(axis=1)/data.shape[1]
# data1.resize([data.shape[0],1])
# data2=np.array([[2],
#               [2],
#               [2]])
# data3=data/data2
# print(data3)
# var = np.array([[0]]*3)
# print(var)
# s=np.dot(data[0],data[0].T)
# print(s)
# R=np.array([[1,0.44,0.29,0.33],
#             [0.44,1,0.35,0.32],
#             [0.29,0.35,1,0.60],
#             [0.33,0.32,0.60,1]])
# val, vector = np.linalg.eig(R)
# print(val.sum())
# print(R[:,0:2])

# data2=np.array([[1,2,3]])
# data3=data2*data
# print(data3)
# data1=np.array([[1,2,3]])
# data2=np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])
# print(data1*data2)





