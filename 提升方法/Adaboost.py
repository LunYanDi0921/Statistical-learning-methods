import pandas as pd
import csv
from datetime import datetime
import numpy as np
import time
import math
import sys
import matplotlib.pyplot as plt
from sklearn import datasets
class Gx:
    def __init__(self,feature,split,coef,left,right):
        #一个分布 这个分布是以哪个feature 作为划分的 划分点 split是 这个分布的系数是coef
         self.feature=feature
         self.split= split
         self.coef=coef
         self.left_class =left#小于切分点的类
         self.right_class = right #大于切分点的类

class Adaboost:
    def __init__(self,Train,Test,M):
        self.Train = Train
        self.Test = Test
        self.w=[1/(self.Train.shape[0])]* self.Train.shape[0] #赋初值,权重的初值是 1/ 训练样本的个数
        self.feature_num = self.Train.shape[1] - 1  # 求特征个数 因为训练数据的最后一列是标签 所以特征数等于维度数减一
        self.label = self.Train.shape[1] - 1  # 标签所在列
        self.Gx = []
        self.M = M  # 迭代次数
        self.split=self.cal_split()

    def cal_split(self):
        # 因为数据的特征都是连续值 计算每一个特征的切分点
        split={}
        for i in range(self.feature_num):
            split[i]=[] #将某一个特征的所有切分点放到一个列表中
            d = self.Train[np.argsort(self.Train[:, i])]  # 以这个特征的大小来进行排序
            for j in range(d.shape[0]-1):
                sp=(d[j][i]+d[j+1][i])/2
                if sp not in split[i]:  #可能有的切分点是重复的 这一步去重
                    split[i].append(sp)
        return split

    def cal_class(self,data):
        #返回数据中正类和负类的占比
        positive =0
        negative =0
        for i in range(data.shape[0]):
            if data[i][self.label]==1:
                positive+=1
            else:
                negative+=1
        return positive/data.shape[0] , negative/data.shape[0]

    def cal_error(self,feature,split,left,right):
       #计算按某个分布的误差
        error=0
        for i in range(self.Train.shape[0]):
            if (self.Train[i][feature] <= split and self.Train[i][self.label]!=left) or (self.Train[i][feature] > split and self.Train[i][self.label]!=right):
               error+=self.w[i]
        return error

    def cal_Gx(self): #求误差最小的分布
        min_error=sys.maxsize #最小误差 李航书上的 em
        min_Gx=None #最小误差对应的分布
        min_a=0
        for i in range(self.feature_num): #遍历每一个特征
            for j in self.split[i]: #遍历切分点
                left = self.Train[(self.Train[:, i] <= j),: ]
                right = self.Train[(self.Train[:, i] > j),: ]
                p1,n1=self.cal_class(left)
                p2,n2=self.cal_class(right)
                if p1>p2:
                    left_class=1
                    right_class=-1
                else:
                    left_class = -1
                    right_class = 1
                error=self.cal_error(i,j,left_class,right_class)
                a = 1/2*math.log((1-error)/error)
                if error < min_error:
                    min_error=error
                    min_Gx=Gx(i,j,a,left_class,right_class)
                    min_a = a
        self.Gx.append(min_Gx)
        return min_a,min_Gx

    def cal_Zm(self,a,Gx):
        #计算规范化因子Zm
        Zm=0
        for i in range(self.Train.shape[0]):
            y=self.Train[i][self.label]
            feature=Gx.feature
            split=Gx.split
            if self.Train[i][feature]<=split:
                g=Gx.left_class
            else:
                g=Gx.right_class
            Zm+= self.w[i]*math.exp(-a*y*g)
        return Zm


    def update_w(self,Zm,a,Gx):
        # 更新权值w
        for i in range(self.Train.shape[0]):
            y = self.Train[i][self.label]
            feature = Gx.feature
            split = Gx.split
            if self.Train[i][feature] <= split:
                g = Gx.left_class
            else:
                g = Gx.right_class
            self.w[i]=self.w[i] * math.exp(-a * y * g) / Zm

    def cal_fx(self,X):
        #计算 f(x)
        fx=0
        for gx in self.Gx:
            feature = gx.feature
            split = gx.split
            left=gx.left_class
            right=gx.right_class
            a=gx.coef
            if X[feature]<=split:
               g = left
            else:
                g=right
            fx +=a*g
        return fx

    def sign_fx(self,fx):
        # 计算 sign(f(x))
        if fx>0:
            return 1
        elif fx<0:
            return -1
        else:
            return 0

    def fit(self):
        for i in range(self.M):
            print(i)
            min_a, min_Gx=self.cal_Gx()
            Zm=self.cal_Zm(min_a,min_Gx)
            self.update_w(Zm,min_a,min_Gx)
            print(self.predict())

    def predict(self):
        right=0
        for i in range(self.Test.shape[0]):
            y=self.sign_fx(self.cal_fx(self.Test[i]))
            if y==self.Test[i][self.label]:
                right+=1
        return right / self.Test.shape[0]

breast_cancer=datasets.load_breast_cancer()
data1=breast_cancer.data
data2=breast_cancer.target
data2.resize([data2.shape[0],1])
for i in range(data2.shape[0]):
    if data2[i][0]==0:
        data2[i][0] =-1
data=np.concatenate((data1,data2), axis=1)
train_size = int(len(data) * 0.7)#划分训练集与测试集
train = np.array(data[:train_size])
# print(X_train.shape)
test=np.array(data[train_size:])
model=Adaboost(train,test,100)
model.fit()
print(model.predict())
