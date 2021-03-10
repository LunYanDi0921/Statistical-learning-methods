import pandas as pd
import csv
from datetime import datetime
import numpy as np
import time
import math
from pandas import read_csv
from sklearn import preprocessing
class LR:#逻辑回归
    def __init__(self,X_Train,Y_train,a,r):
        self.X_Train=X_Train
        self.Y_train=Y_train
        self.a=a
        self.w=self.initialization_parameters() #参数 由(w1,w2,w3 ...，b)组成
        self.r=r #迭代多少次
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def initialization_parameters(self): #初始化 w为一个n+1 维的向量 n是有n维特征
        n=self.X_Train.shape[1]
        w=np.ones((n,1))
        return w
    def cost_function(self): #损失函数
        lost=np.sum(self.Y_train*np.log2(self.sigmoid(np.dot(self.X_Train,self.w)))+(1-self.Y_train)*np.log2(1-self.sigmoid(np.dot(self.X_Train,self.w))))/self.X_Train.shape[0]
        return -lost
    def cal_dw(self): #求w的偏导数
        dw=np.dot((self.sigmoid(np.dot(self.X_Train,self.w))-self.Y_train).T,self.X_Train)/self.X_Train.shape[0]
        return dw.T
    def fit(self):
         for i in range(self.r): #循环self.r 次
             print("损失值为:")
             print(self.cost_function())
             self.w=self.w-self.a*self.cal_dw()
    def predict(self,X_Test,Y_Test):
        right = 0
        for i in range(X_Test.shape[0]):
            p=self.sigmoid(np.dot(X_Test[i],self.w))
            q=1-p
            if p>q:
                if Y_Test[i][0]==1:
                    right+=1
            else:
                if Y_Test[i][0]==0:
                    right+=1
        print("准确率为")
        print(right/X_Test.shape[0])
def read_data(filename):#将文件中的数据读出
    df = read_csv(filename)
    return df
def prepare_data(df):  # 数据预处理
    ndarray_data = df.values
    X = df.iloc[:, 2:21]  # 数据切片
    Y = df.iloc[:, 21]
    print(X)
   # 特征值标准化\n"
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = minmax_scale.fit_transform(X)
   # Y = Y.replace(0, -1)
    return X, Y
df=read_data("Customer-Churn.csv")#读数据
df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")#TotalCharges中有空格
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
TotalCharges = []
for i in range(df.shape[0]):
    TotalCharges.append(float(df.loc[i, 'TotalCharges']))  # 将字符串数据转化为浮点型加入到数组之中
avg=np.var(TotalCharges)
df['TotalCharges'] = df['TotalCharges'].replace(0, avg)#用均值填充缺失值
print(df.shape)#(7043, 21)
#print(df.dtypes)#查看各列数据的数据类型
X,Y=prepare_data(df)
train_size = int(len(X) * 0.8)#划分训练集与测试集
X_train = np.array(X[:train_size])
print(X_train.shape)
Y_train=np.array(Y[:train_size])
Y_train.resize([Y_train.shape[0],1])
X_test = np.array(X[train_size:])
Y_test =np.array( Y[train_size:])
Y_test.resize([Y_test.shape[0],1])
a = np.ones(X_train.shape[0])
b = np.ones(X_test.shape[0])
X_train=np.insert(X_train, 19, values=a, axis=1)
X_test=np.insert(X_test, 19, values=b, axis=1)
model=LR(X_train,Y_train,0.05,20000)
model.fit()
model.predict(X_test,Y_test)


