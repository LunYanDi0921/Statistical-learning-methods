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
import sys
class Node:
    def __init__(self, isvisit, left,right,X,split,label):#是否访问过,左子树,右子树,特征坐标,切分轴,标签
        self.isvisit = isvisit
        self.left = left
        self.right = right
        self.X=X
        self.split=split
        self.label=label
class KNN:#基于kd树进行k近邻搜索的KNN算法
    def __init__(self,k,data,label):
        self.k=k # 找多少个近邻
        #self.data=data
        self.stack=[]
        self.root=self.create_kdtree(data,0,label) #kd树根结点
    def create_kdtree(self,data,split,label):
        #print("split="+str(split))
        len = data.shape[0]  # 求出datas长度方便找中位数
        if len==0:
            return None
        data = data[np.argsort(data[:, split])]#按照划分的那一维度进行排序
       # print(data)
       # print("******************")
        dimension=data.shape[1]#求出数据维度
        #print("dimension=" +str(dimension))
        print()
        mid=int(len/2) #中位数
        if len%2==0:
            mid=mid-1
        left=data[0:mid,:] #左孩子
        llabel=label[0:mid,:]
        now=data[mid:mid+1,:]#当前结点
        nlabel = label[mid:mid+1,:]
        #print(nlabel[0][0])
        right=data[mid+1:,:] #右孩子
        rlabel= label[mid+1:,:]
        #print("now: ")
       # print(now)
        node=Node(False,None,None,now[0],split,nlabel[0][0])
       # print(node.X)
        node.left=self.create_kdtree(left,(split+1)%dimension,llabel)
        node.right=self.create_kdtree(right,(split+1)%dimension,rlabel)
        return node
    def pre_order(self,node):#先序遍历
        if node!=None:
            print(node.X)
            self.pre_order(node.left)
            self.pre_order(node.right)

    def in_order(self, node):  # 中序遍历
        if node != None:
            self.in_order(node.left)
            print(node.X)
            self.in_order(node.right)
    def search_k_neighbor(self,X): #寻找某个点的k个邻居结点
        L=[]#存放k个邻居结点的列表
        node=self.root #从根节点开始遍历
        Lmax=0
        maxi=0
        node = self.search_leaf_node(node, X)  # 首先寻找叶子节点
        node.isvisit = True #设置为已访问
        Lmax= self.EuclideanDistance(X, node.X)
        L.append(node)
        while len(self.stack)!=0 :
            node=self.stack.pop()
            if node.isvisit==False: #如果这个结点没有被访问过
                node.isvisit=True
                dist=self.EuclideanDistance(X,node.X)
                if len(L)<self.k:#如果L中不足k个结点直接放入
                    if Lmax<dist: #有需要的话调整最大距离
                        Lmax=dist
                        maxi=len(L)
                    L.append(node)
                elif dist<Lmax:#当前点到 X的距离 比L中结点到X的最大距离 要小 说明需要更新L
                     L.pop(maxi)
                     L.append(node)
                     Lmax,maxi= self.max_dist(L,X)
                split=node.split
                if abs(X[split]-node.X[split]) < Lmax:#从另一支出发
                    left=node.left
                    right=node.right
                    if left!=None and left.isvisit==False:
                           node=left
                           node = self.search_leaf_node(node, X)  # 首先寻找叶子节点
                           node.isvisit = True
                           dist=self.EuclideanDistance(X,node.X)
                           if len(L) < self.k:  # 如果L中不足k个结点直接放入
                               if Lmax < dist:
                                   Lmax = dist
                                   maxi = len(L)
                               L.append(node)
                           elif dist < Lmax:  # 当前点到 X的距离 比L中结点到X的最大距离 要小 说明需要更新L
                               L.pop(maxi)
                               L.append(node)
                               Lmax, maxi = self.max_dist(L, X)
                    if right!=None and right.isvisit==False:
                           node=right
                           node = self.search_leaf_node(node, X)  # 首先寻找叶子节点
                           node.isvisit = True
                           dist = self.EuclideanDistance(X, node.X)
                           if len(L) < self.k:  # 如果L中不足k个结点直接放入
                               if Lmax < dist:
                                   Lmax = dist
                                   maxi = len(L)
                               L.append(node)
                           elif dist < Lmax:  # 当前点到 X的距离 比L中结点到X的最大距离 要小 说明需要更新L
                               L.pop(maxi)
                               L.append(node)
                               Lmax, maxi = self.max_dist(L, X)

        return  L
    def max_dist(self,L,X):
        Lmax=0
        maxi=0
        for i in range(len(L)):
            dist=self.EuclideanDistance(L[i].X,X)
            if dist>Lmax:
                Lmax=dist
                maxi=i
        return Lmax,maxi
    def EuclideanDistance(self,X1,X2):#欧氏距离计算公式
        dist=0
        for (x1,x2) in zip(X1,X2):
            dist+=(x1-x2)**2
        return dist**0.5
    def search_leaf_node(self,node,X):
        while (node.left != None) or (node.right != None):  # 不是叶子节点
            self.stack.append(node)
            if node.left == None:  # 单分支结点
                node = node.right
            elif node.right == None:  # 单分支结点
                node = node.left
            else:
                split = node.split  # 以哪个轴做分割的
                nodeX = node.X
                if X[split] > nodeX[split]:
                    node = node.right
                else:
                    node = node.left
        return node
    def predict(self,X_test,Y_test):
        n = X_test.shape[0]  # 共有多少条数据
        num = 0  # 正确预测的数据有多少
        for i in range(X_test.shape[0]):
            X = X_test[i]
            Y = Y_test[i]
            L=self.search_k_neighbor(X)
            positive =0
            negative =0
            for node in L:
               if node.label==1:
                   positive+=1
               else:
                   negative+=1
            if positive>=negative:
                label=1
            else:
                label=-1
            if label==Y[0]:
                num+=1
        return num / n
def read_data(filename):#将文件中的数据读出
    df = read_csv(filename)
    return df
def prepare_data(df):  # 数据预处理
    ndarray_data = df.values
    X = df.iloc[:, 2:21]  # 数据切片
    Y = df.iloc[:, 21]
   # print(X)
    # 特征值标准化\n"
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = minmax_scale.fit_transform(X)
    Y = Y.replace(0, -1)
    return X, Y
'''data = {
        'X':[6.27,1.24,17.05,-6.88,-2.96,7.75,10.80,-4.60,-4.96,1.75,15.31,7.83,14.63],
        'Y':[5.50,-2.86,-12.79,-5.40,-0.50,-22.68,-5.03,-10.55,-12.61,12.26,-13.16,15.70,-0.35]
        }
df = pd.DataFrame(data,columns=['X','Y'])'''
'''data=np.array([[6.27,5.50],[1.24,-2.86],[17.05,-12.79],[-6.88,-5.40],[-2.96,-0.50],[7.75,-22.68],[10.80,-5.03],[-4.60,-10.55],[-4.96,12.61],[1.75,12.26],[15.31,-13.16],[7.83,15.70],[14.63,-0.35]])
#print(data)
data = data[np.argsort(data[:,0])]#按第一列排序
print (data)
data=data[1:,:]
print (data)'''

#data=np.array([[6.27,5.50],[1.24,-2.86],[17.05,-12.79],[-6.88,-5.40],[-2.96,-0.50],[7.75,-22.68],[10.80,-5.03],[-4.60,-10.55],[-4.96,12.61],[1.75,12.26],[15.31,-13.16],[7.83,15.70],[14.63,-0.35]])
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
train_size = int(len(X) * 0.7)#划分训练集与测试集
X_train = np.array(X[:train_size])
print(X_train.shape)
Y_train=np.array(Y[:train_size])
Y_train.resize([Y_train.shape[0],1])
X_test = np.array(X[train_size:])
Y_test =np.array( Y[train_size:])
Y_test.resize([Y_test.shape[0],1])
knn=KNN(10,X_train,Y_train)
print(knn.predict(X_test,Y_test))
#X=np.array([-1,-5])
#L=knn.search_k_neighbor(X)
#for node in L:
   # print(node.X)
#knn.pre_order(knn.root)
print("*********************")
print(X_test[0])
#knn.in_order(knn.root)
#d=data[0:1,:]
#print(d[0][0])
#print(100<sys.maxsize)
'''datas=[]
datas.append(1)
datas.append(2)
datas.append(3)
print(datas.pop()) # 3'''
