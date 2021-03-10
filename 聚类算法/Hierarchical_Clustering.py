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
class Hierarchical_Clutering:
    def __init__(self,Train):
        self.Train=Train
        self.M=self.Train.shape[1] #样本的维数
        self.N=self.Train.shape[0] #样本的个数
        self.D=self.cal_D()
        self.classes,self.class_index=self.generating_classes() #类集合

    def Minkowski_distance(self,X,Y,p):
        #闵可夫斯基距离
        max=0
        d=0
        if p>sys.maxsize: #如果p是无穷 即切比雪夫距离
            for i in range(self.M):
                if math.fabs(X[i]-Y[i]) > max:
                    max=math.fabs(X[i]-Y[i])
            d=max
        else:
            sum=0
            for i in range(self.M):
                sum+=math.pow(math.fabs(X[i]-Y[i]),p)
            d=math.pow(sum,1/p)
        return d

    def cal_D(self):
     #计算n个样本两两之间的欧氏距离
     D=[[0]*self.N]*self.N
     for i in range(self.N):
         for j in range(i+1,self.N):
             d=self.Minkowski_distance(self.Train[i],self.Train[j],2)
             D[i][j]=d
             D[j][i]=d
     return D

    def generating_classes(self):
     #构造n个类，使每个类只包含一个样本
         classes={}
         class_index=[]
         for i in range(self.N):
             classes[i]=[]
             classes[i].append((i,self.Train[i])) #i 表示第几个样本 self.Train[i] 是样本的内容
             class_index.append(i) #类别
         return classes,class_index

    def fit(self):
        lens=len(self.classes)
        min_c1=0
        min_c2=0
        while len(self.classes)>1: #类的总数大于等于2 就可以继续合并
           print(self.classes)
           l=len(self.class_index)
           print(self.class_index)
           min_dis = sys.maxsize
           for i in range(l):
              for j in range(i+1,l):
                  c1_num=self.class_index[i]
                  c2_num=self.class_index[j]
                  class1=self.classes[c1_num]
                  class2=self.classes[c2_num]
                  for c1 in range(len(class1)):
                      for c2 in range(len(class2)):
                           d=self.D[class1[c1][0]][class2[c2][0]]
                           if d <min_dis:
                               min_dis=d
                               min_c1= c1_num
                               min_c2= c2_num
           for c in self.classes[min_c2]:
              #print(c)
              self.classes[min_c1].append(c)
           del self.classes[min_c2]
           self.class_index.remove(min_c2)


train=np.array([[0,2],
                [0,0],
                [1,0],
                [5,0],
                [5,2]])
model=Hierarchical_Clutering(train)
model.fit()







    #def fit(self):






