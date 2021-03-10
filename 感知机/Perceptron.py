import pandas as pd
import csv
from datetime import datetime
import numpy as np
import time
from pandas import Series
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas import read_csv
from pandas import set_option
from pandas import DataFrame
class Model:#感知机模型
    def __init__(self,data):
        self.w = np.zeros(df.columns.size-3, dtype=np.float32)
        self.b=0
        self.a=0.1#学习率
    def sign(self, x, w, b):
        y=np.dot(w,x)+b
        return y
    def fit(self, X_train, Y_train):#训练模型
        isfinish=False
        while isfinish==False:
           isfinish = True
           for i in range(X_train.shape[0]):
               print(i)
               X=X_train[i]
               Y=Y_train[i]
               y=self.sign(X, self.w, self.b)
               if Y*y <=0:#如果该点是误分类点
                   self.w=self.w+self.a * np.dot(Y,X)
                   self.b=self.b+ self.a*Y
                   isfinish = False
        return self.w,self.b #返回训练好的模型的参数
    def predict(self,X_test,Y_test):#测试模型的准确率
        n=X_test.shape[0]#共有多少条数据
        num=0#正确预测的数据有多少
        for i in range(X_test.shape[0]):
            X=X_test[i]
            Y=Y_test[i]
            y=self.sign(X, self.w, self.b)
            if Y*y>0:
                num+=1
        return num/n
def read_data(filename):#将文件中的数据读出
    df = read_csv(filename)
    return df
def datapreprocessing(data):#数据预处理
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'].values)
    data['Partner'] = le.fit_transform(data['Partner'].values)
    data['Dependents'] = le.fit_transform(data['Dependents'].values)
    data['PhoneService'] = le.fit_transform(data['PhoneService'].values)
    data['MultipleLines'] = le.fit_transform(data['MultipleLines'].values)
    data['InternetService'] = le.fit_transform(data['InternetService'].values)
    data['OnlineSecurity'] = le.fit_transform(data['OnlineSecurity'].values)
    data['OnlineBackup'] = le.fit_transform(data['OnlineBackup'].values)
    data['DeviceProtection'] = le.fit_transform(data['DeviceProtection'].values)
    data['TechSupport'] = le.fit_transform(data['TechSupport'].values)
    data['StreamingTV'] = le.fit_transform(data['StreamingTV'].values)
    data['StreamingMovies'] = le.fit_transform(data['StreamingMovies'].values)
    data['Contract'] = le.fit_transform(data['Contract'].values)
    data['PaperlessBilling'] = le.fit_transform(data['PaperlessBilling'].values)
    data['PaymentMethod'] = le.fit_transform(data['PaymentMethod'].values)
    data['Churn'] = le.fit_transform(data['Churn'].values)
    print(data)
    data.to_csv('Customer-Churn.csv')#
def prepare_data(df):#数据预处理
       ndarray_data = df.values
       X=df.iloc[:,2:21]#数据切片
       Y=df.iloc[:,21]
       print(X)
       #特征值标准化\n"
       minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
       X = minmax_scale.fit_transform(X)
       Y = Y.replace(0, -1)
       return X , Y
df=read_data("Customer-Churn.csv")#读数据
df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")#TotalCharges中有空格
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
TotalCharges = []
for i in range(df.shape[0]):
    TotalCharges.append(float(df.loc[i, 'TotalCharges']))  # 将字符串数据转化为浮点型加入到数组之中
avg=np.var(TotalCharges)
df['TotalCharges'] = df['TotalCharges'].replace(0, avg)#用均值填充缺失值
print(df.shape)#(7043, 21)
print(df.dtypes)#查看各列数据的数据类型
X,Y=prepare_data(df)
train_size = int(len(X) * 0.7)#划分训练集与测试集
X_train = X[:train_size]
Y_train= Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]
model=Model(X)
w,b=model.fit(X_train,Y_train)
print("w="+w)
print("b="+b)
print("模型的准确率是: "+model.predict(X_test,Y_test))

