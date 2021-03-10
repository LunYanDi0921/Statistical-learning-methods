import pandas as pd
import csv
from datetime import datetime
import numpy as np
import time
import math
from pandas import Series
import matplotlib.pyplot as plt
class Edge: #代表一个分支
    def __init__(self,  child, value):
        self.child=child #这个边连接的结点
        self.value=value #这个边的值
class Node:
    def __init__(self,data,edges,feature,value):
# data表示这个结点包含的数据,childs表示该结点的孩子结点(因为基于C4.5算法的决策树并不是一棵二叉树,feature表示该结点是以哪个特征来划分子结点的，value是分类值 只有叶结点才有取值
     self.data=data
     self.edges=edges
     self.feature=feature
     self.value=value

class Decision_Tree:#基于C4.5算法的决策树
    def __init__(self,Train,features,label,a,b,feature_name):#features是当前可用特征集初始时是所有特征
        self.train=Train
        self.features=features
        self.label=label#标签所在列
        self.b=b#信息增益比的阈值
        self.a=a #损失函数的参数
        self.T = 0  # 一共有多少叶子结点
        self.CT = 0
        self.feature_name = feature_name
        self.root=self.create_tree(Train)
    def information_entropy(self,data,label):#计算data数据集的信息熵 label表示以哪一列作为求信息熵的基准
        labels=self.class_num(data,label)
        #print(labels)
        ie=0
        for k,v in labels.items():#计算信息熵
            p=v/data.shape[0]
            if p!=0:
              ie+= -p*math.log2(p)
        return ie # 返回信息熵
    def conditional_entropy(self,data,feature): #计算条件熵  feature表示条件熵中的特征的列值 label表示标签的列值
        fv=self.class_num(data,feature)
        datas={}# 根据feature的不同取值划分出多个数据块 存放在datas中
        ce=0
        for k,v in fv.items():
            d=data[(data[:, feature] == k),: ] #把feature那一列取值为k的数据放在一起
            datas[k]=d
        for k,v in datas.items():
            p=fv[k]/data.shape[0]
            ie=self.information_entropy(v,self.label)
            ce+=p*ie
        return ce #返回条件熵 ief 是数据集data 关于特征feature的值的熵
    def information_gain(self,data,feature):# 计算某个特征的信息增益比
        ie=self.information_entropy(data,self.label)
        ce=self.conditional_entropy(data,feature)
        ief=self.information_entropy(data,feature)
        # print(ie)
        # print(ce)
        ig=ie-ce
        # if ief !=0:
        #   return  ig / ief
        # else:
        return ig

    def processing_continuous_values(self,data,feature):#对连续值进行处理
        data = data[np.argsort(data[:, feature])]  # 将data 按照 feature这一列排序
        X = data[:, feature:feature + 1] #把这一列切分
        splits=[] #切分点
        ie= self.information_entropy(data,self.label)
        max_ig=0
        max_split=0
        ief=0
        for i in range(X.shape[0]-1):
            split=(X[i][0]+X[i+1][0])/2
            splits.append(split)
        for i in range(len(splits)):#让splits中的每一个点s依次做切分点 将数据分为 < s 和 > s
            d1 = data [(data[:, feature] <splits[i]),:]
            d2 = data [(data[:, feature] > splits[i]),:]
            ie1=self.information_entropy(d1, self.label)
            ie2 =self.information_entropy(d2, self.label)
            p1=d1.shape[0]/X.shape[0]
            p2=d2.shape[0]/X.shape[0]
            ce=p1*ie1+p2*ie2
            if p1>0:
              ief+=-p1*math.log2(p1)
            if p2>0:
              ief += -p2 * math.log2(p2)
            if (ie-ce)/ief> max_ig:
                max_ig=(ie-ce)/ief
                max_split=splits[i]
        return max_ig,max_split
    def is_one_class(self,data,label):#判断data中的数据是否属于同一类 并返回类数组
        X=data[:, label:label + 1]
        labels=[]
        for i in  range(X.shape[0]):
            if X[i][0] not in labels:
                labels.append(X[i][0])
        if  len(labels) == 1:
            return True
        else:
            return False
    def max_num_class(self,data,label):# 返回取值最多的那一类
        #X = data[:, label:label + 1]
        labels = self.class_num(data,label)
        max_num=0
        max_class = 0
        for k,v in labels.items():
            if v > max_num:
                max_num=v
                max_class=k
        return max_class
    def class_num(self,data,feature):# 对于某个特征而言 他有多少种取值
        X = data[:, feature:feature + 1]
        fea_values = {}
        for i in range(X.shape[0]):
            #print(X[i])
            if X[i][0] not in fea_values:
                fea_values[X[i][0]]=1
            else:
                fea_values[X[i][0]] += 1
        return  fea_values
    def cal_leaf(self,data):#计算data数据集的信息熵 label表示以哪一列作为求信息熵的基准
        labels=self.class_num(data,self.label)
        leaf_ie=0
        for k,v in labels.items():
            p=v/data.shape[0]
            if p!=0:
              leaf_ie+= -v*math.log2(p)
        return leaf_ie
    def cost_function(self,CT,T):
        return CT+self.a*T
    def create_tree(self,data):# 生成一棵决策树 label
        node=None
        max_ig=0 #利用哪个特征进行划分
        max_feature=0
        if self.is_one_class(data,self.label) : #如果数据都属于一类
            node=Node(data,None,self.label,self.max_num_class(data,self.label))#叶子结点
            self.T +=1
            self.CT+=self.cal_leaf(data)
        elif len(self.features)==0: #没有可供划分的特征了
            node = Node(data, None, self.label, self.max_num_class(data, self.label)) #叶子结点
            self.T += 1
            self.CT += self.cal_leaf(data)
        else:
            for i in self.features: #对每一个特征计算信息增益比
                # if i==4 or i==17 or i==18: #如果该特征是连续型特征
                #     ig=self.processing_continuous_values(data,i)
                # else:
                ig=self.information_gain(data,i)
                # print(self.feature_name[i])
                # print(ig)
                # print("**********")
                if ig > max_ig:
                    max_ig = ig
                    max_feature = i
            if max_ig<self.b:
                node = Node(data, None, self.label, self.max_num_class(data, self.label))  # 叶子结点
                self.T += 1
                self.CT += self.cal_leaf(data)
            else:
                self.features.remove(max_feature)
                edges=[]#子树列表
                fea_values=self.class_num(data,max_feature)
                for k in fea_values.keys():
                   d=data [(data[:, max_feature] == k),:] #按所选特征的不同取值来划分数据
                   child=self.create_tree(d)
                   edge=Edge(child,k)
                   edges.append(edge)
                node=Node(data,edges,max_feature,None)
        return node
    def pre_order(self,node):
        if node!=None:
            if node.edges==None: #如果是叶子结点
               print(node.value+"\n")
            else:
                print(self.feature_name[node.feature])
                for edge in node.edges:
                    print(edge.value)
                    self.pre_order(edge.child)

    def level_order(self):
        queue=[]
        queue.append(self.root)
        while len(queue) > 0:
            node= queue.pop(0)
            if node.edges == None:
                print(node.value)
            else:
                print(self.feature_name[node.feature])
                for edge in node.edges:
                   print(edge.value)
                   queue.append(edge.child)
    def prune(self,node): #后剪枝
        child_CT=0 #子树上的CT 值
        child_T=0 #子树上的叶子结点个数
        if node.edges==None: #如果是叶子结点
            return self.cal_leaf(node.data),1
        else :
            for edge in node.edges:
                child=edge.child
                ct,t=self.prune(child)
                child_CT+=ct
                child_T+=t
            self_CT=self.cal_leaf(node.data) #如果将该结点作为叶子结点它的CT值
            if self.cost_function(self.CT-child_CT+self_CT,self.T-child_T+1)<self.cost_function(self.CT,self.T): #需要剪枝
                node.edges=None
                self.CT=self.CT-child_CT+self_CT
                self.T=self.T-child_T+1
                node.feature= self.label
                node.value=self.max_num_class(node.data,self.label)
                return self_CT,1
            else: #不需要剪枝
                return child_CT,child_T
# Train  = np.array([[0,0,0,0,0,0,1],
#                    [1,0,1,0,0,0,1],
#                    [1,0,0,0,0,0,1],
#                    [0,1,0,0,1,1,1],
#                    [1,1,0,1,1,1,1],
#                    [0,2,2,0,2,1,0],
#                    [2,1,1,1,0,0,0],
#                    [1,1,0,0,1,1,0]
#                    [2,0,0,2,2,0,0]
#                    [0,0,1,1,1,0,0]])
# Train= np.array([['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
#                  ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
#                  ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
#                  ['青绿','稍蜷','浊响','清晰','稍凹','软粘','是'],
#                  ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
#                  ['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
#                  ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
#                  ['乌黑','稍蜷','浊响','清晰','稍凹','软粘','否'],
#                  ['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
#                  ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']
#                 ])
# features=[4,1,2,3,0,5]
# feature_name=['色泽','根蒂','敲声','纹理','脐部','触感']
# label=6
# DT=Decision_Tree(Train,features,label,0.5,0,feature_name)
# #DT.order(DT.root)
# DT.pre_order(DT.root)
# print("*******")
# DT.level_order()
# #DT.prune(DT.root)
# #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# #DT.order(DT.root)
# p=3/5
# print(-p*math.log2(p))
datasets = np.array([['青年', '否', '否', '一般', '否'],
                     ['青年', '否', '否', '好', '否'],
                     ['青年', '是', '否', '好', '是'],
                     ['青年', '是', '是', '一般', '是'],
                     ['青年', '否', '否', '一般', '否'],
                     ['中年', '否', '否', '一般', '否'],
                     ['中年', '否', '否', '好', '否'],
                     ['中年', '是', '是', '好', '是'],
                     ['中年', '否', '是', '非常好', '是'],
                     ['中年', '否', '是', '非常好', '是'],
                     ['老年', '否', '是', '非常好', '是'],
                     ['老年', '否', '是', '好', '是'],
                     ['老年', '是', '否', '好', '是'],
                     ['老年', '是', '否', '非常好', '是'],
                     ['老年', '否', '否', '一般', '否'],
                     ['青年', '否', '否', '一般', '是']])
#
features=[0,1,2,3]
feature_name=['年龄','有工作','有自己房子','信用情况']
label=4
DT=Decision_Tree(datasets,features,label,0.5,0,feature_name)
#DT.order(DT.root)
DT.pre_order(DT.root)
print("*******")
DT.level_order()

DT.prune(DT.root)
DT.pre_order(DT.root)
print("*******")
DT.level_order()









