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
class Maximum_Entropy:
    def __init__(self,X_Train,Y_Train,X_Test,Y_Test,iteration):
        self.X_Train=X_Train #训练集特征
        self.Y_Train=Y_Train  #训练集标签
        self.X_Test=X_Test #测试集特征
        self.Y_Test=Y_Test #测试集标签
        self.feature_num = X_Train.shape[1]
        self.n = 0  # 一共有多少个特征函数
        self.fi = self.cal_fi()
        self.w=[0]*self.n
        self.sigma=[0.5]*self.n
        self.fi_index,self.index_fi,self.index_feature= self.create_fi_map()
        self.iteration=iteration

    def cal_fi(self):
        #在本方法中为每一个特征都建一个字典 并把这些字典存放到fi列表中
        #每个特征的字典 的键是这个特征的某个取值x 以及这个取值所在的那个样本的标签y 所组成的元组(x,y) 它的value是在训练集中(x,y) 出现了多少次
        #一个元组(x,y)代表的就是一个特征函数 fi
        fi=[]
        for i in range(self.feature_num): # fi 中是 feature_num个 字典
            fi.append({})
        for i in range(self.X_Train.shape[0]):#遍历所有数据
            for j in range(self.feature_num):
                if (self.X_Train[i][j],self.Y_Train[i][0]) not in fi[j]:
                     fi[j][(self.X_Train[i][j],self.Y_Train[i][0])]=1
                else:
                     fi[j][(self.X_Train[i][j], self.Y_Train[i][0])] += 1
        for dict in fi:
            self.n+=len(dict)

        return fi

    def create_fi_map(self):
        #本方法是给self.fi中的所有"特征函数" 赋一个索引值的 为了之后方便查找
        index=0
        fi_index={}
        index_fi=[0]*self.n
        index_feature=[0]*self.n
        for i in range(self.feature_num):
            for (x,y)  in self.fi[i]:
                fi_index[(x,y)]=index
                index_fi[index]=(x,y)
                index_feature[index]=i
                index+=1
        return fi_index,index_fi,index_feature

    def cal_class(self):
        #本方法是计算数据一共被分为几类 每一类是如何表示的
        list=[]
        for i in range(self.Y_Train.shape[0]):
            if self.Y_Train[i][0] not in list:
                list.append(self.Y_Train[i][0])
        return list

    def cal_p(self,X,Y):
    #本方法是求解最大熵模型的，对应于李航《统计学习方法》 公式6.22
        sum=0
        for i in range(self.feature_num):
            if (X[i],Y) in self.fi[i]:
               index=self.fi_index[(X[i],Y)]
               sum+=self.w[index]
       # print(sum)
        e1=math.exp(sum)
        label=self.cal_class()
        label.remove(Y)
        e2=0
        for y in label:
           sum=0
           for i in range(self.feature_num):
             if (X[i],y) in self.fi[i]:
               index=self.fi_index[(X[i],y)]
               sum+=self.w[index]
              # print(self.w)
              # print(sum)
           e2+=math.exp(sum)
        return e1/(e1+e2)

    def cal_E_P(self):
        #计算特征函数关于经验分布P(X,Y)的期望值
        ep=[0]*self.n
        for i in range(self.feature_num):
            for (x,y) in self.fi[i]:
                index=self.fi_index[(x,y)]
                ep[index] = self.fi[i][(x,y)]/self.X_Train.shape[0]
        return ep
    def fjing(self,X,Y):# f#(x,y)
        sum=0
        for i in range(self.feature_num):
            if (X[i],Y) in self.fi[i]:
                sum+=1
        return sum
    def cal_px(self,feature,x_value):
        X = self.X_Train[:, feature:feature + 1]
       # print(X)
        dict={}
        for i in range(X.shape[0]):
            if X[i][0] not in dict:
                dict[X[i][0]]=1
            else:
                dict[X[i][0]]+=1
        return dict[x_value] / X.shape[0]

    def gsigma(self,index):
        ep=self.cal_E_P()
        p1=ep[index]
        feature=self.index_feature[index]
        sigma=self.sigma[index]
        label=self.cal_class()
        p2=0
        for i in range(self.X_Train.shape[0]):
            inner_sum=0
            for y in label:
                if (self.X_Train[i][feature],y) == self.index_fi[index]: #满足特征函数i
                   inner_sum+= self.cal_p(self.X_Train[i],y)*math.exp(sigma*self.fjing(self.X_Train[i],y))
            p2+=self.cal_px(feature,self.X_Train[i][feature])*inner_sum
        return p1-p2

    def cal_gsigma_derivatives(self,index):
        feature = self.index_feature[index]
        sigma = self.sigma[index]
        label = self.cal_class()
        p = 0
        for i in range(self.X_Train.shape[0]):
            inner_sum = 0
            for y in label:
                if (self.X_Train[i][feature], y) == self.index_fi[index]:  # 满足特征函数i
                    inner_sum += self.cal_p(self.X_Train[i], y)*self.fjing(self.X_Train[i], y)* math.exp(sigma * self.fjing(self.X_Train[i], y))
            p += self.cal_px(feature, self.X_Train[i][feature]) * inner_sum
        return -p
    def fit(self):#训练模型 参数的设置
        for i in range(self.iteration):
            print(i)
            for j in range(self.n):
                self.sigma[j] = self.sigma[j]-(self.gsigma(j)/self.cal_gsigma_derivatives(j))
                self.w[j] = self.w[j] + self.sigma[j]
            print(self.predict())
    def predict(self):
        right=0
        label=self.cal_class()
        for i in range(self.X_Test.shape[0]):
            max = 0
            maxy = 0
            for y in label:
                p=self.cal_p(self.X_Test[i],y)
                if max<p:
                   max=p
                   maxy=y
            if maxy==self.Y_Test[i][0]:
                right+=1
        return right/self.X_Test.shape[0]

def read_data(filename):#将文件中的数据读出
    df = read_csv(filename)
    return df
def prepare_data(df):  # 数据预处理
    X = df[:, 0:16]  # 数据切片
    Y = df[:, 16]
    # 特征值标准化\n"
    #minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #X = minmax_scale.fit_transform(X)
   # Y = Y.replace(0, -1)
    for i in range(X.shape[0]):
         X[i][0] =(int)(X[i][0]/10)
    # print(X)
    # print(Y)
    return X, Y
df=read_csv('diabetes_data_upload.csv')
data=np.array(df)
X,Y=prepare_data(data)
train_size = int(len(X) * 0.7)#划分训练集与测试集
X_train = np.array(X[:train_size])
# print(X_train.shape)
Y_train=np.array(Y[:train_size])
Y_train.resize([Y_train.shape[0],1])
# print(X_train[0][0])
# print(Y_train[2][0])
X_test = np.array(X[train_size:])
Y_test =np.array( Y[train_size:])
Y_test.resize([Y_test.shape[0],1])
model=Maximum_Entropy(X_train,Y_train,X_test,Y_test,100)
model.fit()
print(model.predict())











