import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas import read_csv
from pandas import DataFrame
import math
class Bayes:
     def __init__(self,X_train,Y_train,X_test,Y_test):
         self.X_train=X_train
         self.Y_train=Y_train
         self.X_test=X_test
         self.Y_test=Y_test

     def calculate_prior_probability(self,c):#计算先验概率 计算 Y=c的先验概率 Y是标签集合
         N = self.Y_train.shape[0]
         label=self.classification(self.Y_train)
         num=label[c]
         return num/N

     def classification(self,Y):
         N = Y.shape[0]  # 一共有多少条数据
         label = {}#用一个列表存储每种分类的数据共有多少条
         for i in range(N):
             if Y[i][0] in label:
                 label[Y[i][0]] += 1
             else:
                 label[Y[i][0]] = 1
         return label

     def fit(self,X):#预测X的标签
          label=self.classification(self.Y_train)
          classs=label.keys() # 返回键的集合 都有哪些标签
          list={}
          max=0
          maxc=0
          for c in classs:
              p1=self.discrete_features(X[0],c,0)
              p2 =self.discrete_features( X[1], c, 1)
              p3=self.discrete_features( X[2], c, 2)
              p4 = self.discrete_features( X[3], c, 3)
              p5 = self.continuous_feature( X[4], c, 4)
              p6 = self.discrete_features( X[5], c, 5)
              p7 = self.discrete_features(X[6], c, 6)
              p8 = self.discrete_features( X[7], c, 7)
              p9 = self.discrete_features(X[8], c, 8)
              p10 = self.discrete_features( X[9], c, 9)
              p11 = self.discrete_features( X[10], c, 10)
              p12 = self.discrete_features( X[11], c, 11)
              p13 = self.discrete_features( X[12], c, 12)
              p14 = self.discrete_features( X[13], c, 13)
              p15 = self.discrete_features( X[14], c, 14)
              p16 = self.discrete_features( X[15], c, 15)
              p17 = self.discrete_features(X[16], c, 16)
              p18 = self.continuous_feature(X[17], c, 17)
              p19 = self.continuous_feature(X[18], c, 18)
              probability=p1*p2*p3*p4*p5*p6*p7*p8*p9*p10*p11*p12*p13*p14*p15*p16*p17*p18*p19*self.calculate_prior_probability(c)
              list[c]= probability
          for k,v in list.items():
              if v>max:
                  max=v
                  maxc=k
          return maxc

     def predict(self):  # 测试模型的准确率
         n = self.X_test.shape[0]  # 共有多少条数据
         num = 0  # 正确预测的数据有多少
         for i in range(self.X_test.shape[0]):
             print(i)
             X = self.X_test[i]
             Y = self.Y_test[i]
             c= self.fit(X)
             if c==Y[0]:
                 num += 1
         return num / n

     def discrete_features(self,v,c,col):#离散类型特征 特征值 标签值 col表明是哪个特征
         m=self.X_train.shape[0]
         label=self.classification(self.Y_train)
         N=label[c] #这个标签的数据共有多少条
         num=0# label=c 且 第col特征 取值为v的数据共有多少条
         classs={} # 对于第col特征来说 它的不同取值都有什么 分别有多少条数据
         for i in range(m):
             if self.X_train[i][col] in classs:
                 classs[self.X_train[i][col]]+=1
             else:
                 classs[self.X_train[i][col]] =1
             if self.X_train[i][col]==v and self.Y_train[i]==c:
                 num+=1
         b=len(classs)
        # print(b)
         return (num+1)/(N+b) #使用拉普拉斯平滑
     def continuous_feature(self,v,c,col):#连续类型特征 特征值 标签值 col表明是哪个特征
         m = self.X_train.shape[0]
         list=[]
         sum=0
         for i in range(m):
             if self.Y_train[i][0]==c:
                 sum+=self.X_train[i][col]
                 list.append(self.X_train[i][col])
         avg=sum/len(list)#求均值
         sum=0
         for i in range(len(list)):
             sum+=(list[i]-avg)**2
         var=sum/len(list)# 求方差
         probability=np.exp(-(v - avg) ** 2 / (2 * var** 2)) / (math.sqrt(2 * math.pi) * var)
         return probability
def read_data(filename):#将文件中的数据读出
    df = read_csv(filename)
    return df
def prepare_data(df):  # 数据预处理
    ndarray_data = df.values
    X = df.iloc[:, 2:21]  # 数据切片
    Y = df.iloc[:, 21]
    print(X)
    # 特征值标准化\n"
    #minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #X = minmax_scale.fit_transform(X)
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
print(X_train[0][0])
print(Y_train[2][0])
X_test = np.array(X[train_size:])
Y_test =np.array( Y[train_size:])
Y_test.resize([Y_test.shape[0],1])
model=Bayes(X_train,Y_train,X_test,Y_test)
print(model.predict())

