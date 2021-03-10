import pandas as pd
import csv
from datetime import datetime
import numpy as np
import time
import math
import sys
from pandas import Series
import matplotlib.pyplot as plt
from sklearn import datasets
class Gaussian_EM:
    def __init__(self,Train,K,M):
        # Train 表示观测变量
        self.Train = Train
        # K表示高斯混合模型有多少分量
        self.K=K
        #a 表示高斯混合模型中每一个模型前面的系数
        self.a=[0.4,0.5,0.1]
        #avg,var=self.init_e_and_var()
        # var 表示高斯混合模型中每一个模型的方差
        self.var=[1000,500,100]
        #u 表示高斯混合模型中每一个模型的期望
        self.u =[16,28,45]
        #迭代次数
        self.M=M
        #第j 个观测来自第k个分模型的概率
        self.gama=None


    def init_e_and_var(self):
        #初始化期望方差
        sum=0
        for i in range(self.Train.shape[0]):
            sum+=self.Train[i][0]
        avg=sum/self.Train.shape[0]
        sum=0
        for i in range(self.Train.shape[0]):
             sum+= (self.Train[i][0]-avg)**2
        var=sum/self.Train.shape[0]
        return avg,var

    def cal_Gaussian_distribution_density(self,y,u,var):
        #计算高斯分布密度
         return 1/math.sqrt(2*math.pi*var)*math.exp(-((y-u)**2)/(2*var))

    def cal_p_of_per_model(self):
        # 计算在当前模型参数下第j 个观测来自第k个分模型的概率
        p=[[0]*self.K]*self.Train.shape[0]
        for j in range(self.Train.shape[0]):
            y = self.Train[j][0]
            for k in range(self.K):
                p1 = self.a[k]*self.cal_Gaussian_distribution_density(y,self.u[k],self.var[k])
                p2 = 0
                for i in range(self.K):
                   p2 +=self.a[i]*self.cal_Gaussian_distribution_density(y,self.u[i],self.var[i])
                   #print(self.cal_Gaussian_distribution_density(y,self.u[i],self.var[i]))
                p[j][k]=p1/p2
        return p

    def cal_u(self):
        #计算在当前参数下期望是多少
        for k in range(self.K):
            p1=0
            p2=0
            for j in range(self.Train.shape[0]):#遍历所有样本点
                p1+= self.Train[j][0]*self.gama[j][k]
                p2+=self.gama[j][k]
            self.u[k]=p1/p2

    def cal_var(self):
        #计算在当前参数下方差的值
        for k in range(self.K):
            p1 = 0
            p2 = 0
            for j in range(self.Train.shape[0]):  # 遍历所有样本点
                p1 +=self.gama[j][k]*((self.Train[j][0]-self.u[k])**2)
                p2 +=self.gama[j][k]
            self.var[k] = p1 / p2

    def cal_a(self):
        #计算在当前模型参数下 每一个分模型系数的值
        for k in range(self.K):
            p = 0
            for j in range(self.Train.shape[0]):  # 遍历所有样本点
                p += self.gama[j][k]
            self.a[k] = p / self.Train.shape[0]

    def cal_max_likehood(self,j):
        # 根据计算第j个观测来自第k个模型的概率，我们可以选择概率最大的模型，来作为第j个观测来自的模型
        #该方法就是计算第j个观测来自的模型
        model=self.gama[j]
        max=0
        maxk=0
        for k in range(len(model)):
            if model[k]> max:
                max=model[k]
                maxk=k
        return maxk

    def cal_likelihood_function(self):
        #计算似然函数
        function_value=0
        for k in range(self.K):
            nk=0
            sum=0
            for j in range(self.Train.shape[0]):
                if self.cal_max_likehood(j)==k:#如果第j个观测是来自k模型的
                    nk+=1
                    sum+=math.log(1/math.sqrt(2*math.pi))-math.log(math.sqrt(self.var[k]))-1/(2*self.var[k])*((self.Train[j][0]-self.u[k])**2)
            function_value+=nk*math.log(self.a[k])+ sum
        return function_value

    def fit(self):
        for i in range(self.M):
            print(i)
            self.gama = self.cal_p_of_per_model()
            self.cal_u()
            self.cal_var()
            self.cal_a()
            print(self.cal_likelihood_function())

Train=np.array([[-67],
                [-48],
                [6],
                [8],
                [14],
                [16],
                [23],
                [24],
                [28],
                [29],
                [41],
                [49],
                [56],
                [60],
                [75]
])
model=Gaussian_EM(Train,3,1000)
model.fit()



