import numpy as np
import math
from sklearn import datasets
import sys
class SVM:
   def __init__(self,X_train,Y_train,X_test,Y_test,kernel,d,sigma,C,e):
       # 训练集特征向量
       self.X_train=X_train
       # 训练集标签
       self.Y_train=Y_train
       # 测试集特征向量
       self.X_test=X_test
       # 测试集标签
       self.Y_test=Y_test
       # 拉格朗日乘子
       self.a=np.array(self.init_a())
       # 偏置 b
       self.b=0
       # 核函数
       self.kernel=kernel
       # 多项式核函数的参数 d
       self.d=d
       # 高斯核函数参数
       self.sigma=sigma
       # 惩罚参数
       self.C = C
       #选择a2使目标函数下降必须大于e
       self.e=e
       # 预测值与真实值的误差Ei
       self.Ei = self.cal_Ei()

   def init_a(self):
       #初始化a a的每一个元素初始化为一个二值列表 第一个值是a[i]的大小 第二个值是a[i] 的下标i 因为之后可能涉及到切片 打乱a的下标
       a=[]
       for i in range(self.X_train.shape[0]):
           a.append([0,i])
       return a

   def select_kernel(self,x1,x2):
       #实现核函数
        if self.kernel=='linear':
            return self.linear_kernel(x1,x2)
        elif self.kernel=='poly':
          return self.poly_kernel(x1,x2)
        else:
          return self.rbf_kernel(x1,x2)

   def linear_kernel(self, x1, x2):
       #实现线性核函数
       return np.dot(x1.T, x2)

   def poly_kernel(self,x1,x2):
        #实现多项式核函数
        return np.power(np.dot(x1.T, x2) + 1, self.d)

   def rbf_kernel(self,x1,x2):
       #实现高斯核函数
       return np.exp(-(self.euclidean_distance(x1, x2) ** 2) / (2*self.sigma ** 2))

   def euclidean_distance(self,x1,x2):
       #实现欧氏距离公式
       sum=0
       for i in range(x1.shape[0]):
           sum += np.power(x1[i]-x2[i],2)
       return np.sqrt(sum)

   def cal_gx(self,x):
       #计算样本x的预测值 g(x)
       gx=0
       #d =  self.a[(self.a[:, 0] > 0 and self.a[:, 0] < self.C), :]
       #left = self.Train[(self.Train[:, i] <= j), :]
       d = self.a[(self.a[:, 0] > 0), :]
       d=d[(d[:,0]<self.C),:]
       for i in range(d.shape[0]):
             gx+= self.d[i][0]*self.Y_train[d[i][1]][0]*self.select_kernel(self.X_train[d[i][1]],x)
       gx+=self.b
       return gx

   def cal_Ei(self):
       # 计算Ei的初始值
       Ei=[0]*self.X_train.shape[0]
       for i in range(self.X_train.shape[0]):
           Ei[i]=self.cal_gx(self.X_train[i])-self.Y_train[i][0]
       return Ei

   def  satisfy_kkt(self,i):
       # 检查第i个样本点是否满足kkt条件
        if self.a[i][0] ==0 and self.Y_train[i][0]*self.cal_gx(self.X_train[i])>=1:
             return True
        elif (self.a[i][0] >0 and self.a[i][0]<self.C ) and self.Y_train[i][0]*self.cal_gx(self.X_train[i])==1:
             return True
        elif self.a[i][0]==self.C and self.Y_train[i][0]*self.cal_gx(self.X_train[i])<=1:
             return True
        else:
             return False

   def target_function(self,index1,index2):
       #计算李航统计学习方法7.101式 目标函数
       W=1/2*self.select_kernel(self.X_train[index1],self.X_train[index1])*(self.a[index1][0]**2)
       W+=1/2*self.select_kernel(self.X_train[index2],self.X_train[index2])*(self.a[index2][0]**2)
       W+=self.Y_train[index1][0]*self.Y_train[index2][0]*self.select_kernel(self.X_train[index1],self.X_train[index2])*self.a[index1][0]*self.a[index2][0]
       W -=self.a[index1][0]
       W -= self.a[index2][0]
       for i in range(self.X_train.shape[0]):
          if i != index1 and i != index2:
              W+=self.Y_train[index1][0]*self.a[index1][0]*self.Y_train[i][0]*self.a[i][0]*self.select_kernel(self.X_train[index1],self.X_train[i])
              W += self.Y_train[index2][0] * self.a[index2][0] * self.Y_train[i][0] * self.a[i][0] * self.select_kernel(self.X_train[index2], self.X_train[i])
       return W

   def select_variable(self):
       #smo算法选择第一个变量
       # 间隔边界上的支持向量
       #data1=self.a[(self.a[:, 0] >0 and self.a[:, 0] <self.C), :]
       data1 = self.a[(self.a[:, 0] > 0), :]
       data1 = data1[(data1[:, 0] < self.C), :]
       # 其余样本点
       data2_1 = self.a[(self.a[:, 0] == 0), :]
       data2_2 =self.a[(self.a[:, 0] == self.C),:]
       data2 = np.concatenate((data2_1, data2_2), axis=0)
       index_1=0
       index_2=0
       a1_old=0
       a2_old=0
       for i in range(data1.shape[0]):
           #首先检查所有间隔边界上的支持向量是否满足kkt条件
           # 发现不满足kkt条件的支持向量
           if self.satisfy_kkt(data1[i][1]) == False:
               # true 代表仍有不满足kkt条件的变量 self.data1[i][1]是它的下标
                index_1=data1[i][1]
                index_2=self.select_second_variable(index_1) #得到第二个变量
                W1=self.target_function(index_1,index_2)
                a1_new, a2_new, a1_old, a2_old=self.cal_update_a1_a2(index_1,index_2)
                self.a[index_1][0] = a1_new  # 更新其值
                self.a[index_2][0] = a2_new
                W2=self.target_function(index_1,index_2)
                self.a[index_1][0] = a1_old  # 更新其值
                self.a[index_2][0] = a2_old
                if W1-W2>self.e:
                    return True,index_1,index_2
                else: #更新失败
                    for j in range(data1.shape[0]): #遍历间隔边界上的支持向量依次作为a2
                        if index_1!=data1[j][1]:
                            index_2=data1[j][1]
                            W1 = self.target_function(index_1, index_2)
                            a1_new, a2_new, a1_old, a2_old = self.cal_update_a1_a2(index_1, index_2)
                            self.a[index_1][0] = a1_new  # 更新其值
                            self.a[index_2][0] = a2_new
                            W2 = self.target_function(index_1, index_2)
                            self.a[index_1][0] = a1_old  # 更新其值
                            self.a[index_2][0] = a2_old
                            if W1 - W2 > self.e:
                                return True,index_1, index_2
                    for j in range(data2.shape[0]):  # 遍历整个训练集依次作为a2
                        if index_1!=data2[j][1]:
                            index_2 =data2[j][1]
                            W1 = self.target_function(index_1, index_2)
                            a1_new, a2_new, a1_old, a2_old = self.cal_update_a1_a2(index_1, index_2)
                            self.a[index_1][0] = a1_new  # 更新其值
                            self.a[index_2][0] = a2_new
                            W2 = self.target_function(index_1, index_2)
                            self.a[index_1][0] = a1_old  # 更新其值
                            self.a[index_2][0] = a2_old
                            if W1 - W2 > self.e:
                                return True,index_1, index_2

       for i in range(data2.shape[0]):
           # 如果所有的间隔边界上的支持向量都满足kkt条件 遍历整个训练集检查它们是否满足kkt条件
           # 发现不满足kkt条件的样本点
           if self.satisfy_kkt(data2[i][1]) == False:
               # true 代表仍有不满足kkt条件的变量 self.data1[i][1]是它的下标
               index_1 = data2[i][1]
               index_2 = self.select_second_variable(index_1)  # 得到第二个变量
               W1 = self.target_function(index_1, index_2)
               a1_new, a2_new, a1_old, a2_old = self.cal_update_a1_a2(index_1, index_2)
               self.a[index_1][0] = a1_new  # 更新其值
               self.a[index_2][0] = a2_new
               W2 = self.target_function(index_1, index_2)
               self.a[index_1][0] = a1_old  # 更新其值
               self.a[index_2][0] = a2_old
               if W1 - W2 > self.e:
                   return True, index_1, index_2
               else:  # 更新失败
                   for j in range(data1.shape[0]):  # 遍历间隔边界上的支持向量依次作为a2
                       if index_1 != data1[j][1]:
                           index_2 = data1[j][1]
                           W1 = self.target_function(index_1, index_2)
                           a1_new, a2_new, a1_old, a2_old = self.cal_update_a1_a2(index_1, index_2)
                           self.a[index_1][0] = a1_new  # 更新其值
                           self.a[index_2][0] = a2_new
                           W2 = self.target_function(index_1, index_2)
                           self.a[index_1][0] = a1_old  # 更新其值
                           self.a[index_2][0] = a2_old
                           if W1 - W2 > self.e:
                               return True, index_1, index_2
                   for j in range(data2.shape[0]):  # 遍历整个训练集依次作为a2
                       if index_1 != data2[j][1]:
                           index_2 = data2[j][1]
                           W1 = self.target_function(index_1, index_2)
                           a1_new, a2_new, a1_old, a2_old = self.cal_update_a1_a2(index_1, index_2)
                           self.a[index_1][0] = a1_new  # 更新其值
                           self.a[index_2][0] = a2_new
                           W2 = self.target_function(index_1, index_2)
                           self.a[index_1][0] = a1_old  # 更新其值
                           self.a[index_2][0] = a2_old
                           if W1 - W2 > self.e:
                               return True, index_1, index_2
               # 表示所有的变量都满足kkt条件
       return False,-1,-1

   def select_second_variable(self,index_1):
       #smo算法选择第二个变量
       #select_first,index_1=self.select_first_variable()
       max_absolute_E = 0
       maei=0
       max_E=0
       maxei=0
       min_E=sys.maxsize
       minei=0
           # 如果已经选择了第一个变量
       E1 = self.Ei[index_1]
       for i in range(len(self.Ei)):
               # 遍历self.Ei 选取最大的Ei 最小的Ei 绝对值最大的Ei 并且记录它们的下标
            if i!=index_1:
                   #保证第二个变量和第一个变量不是相同的
                ae=math.fabs(self.Ei[i])
                if ae>max_absolute_E:
                     max_absolute_E=ae
                     maei=i
                if self.Ei[i]>max_E:
                     max_E=self.Ei[i]
                     maxei=i
                if self.Ei[i]<min_E:
                     min_E=self.Ei[i]
                     minei=i
       if E1==0:
               #如果E1等于0 就返回绝对值最大的Ei的下标
            return maei
       elif E1>0:
               #如果E1>0 返回值最小的Ei的下标
            return minei
       else:
               #如果E1<0 返回值最大的Ei的下标
            return  maxei

   def cal_boundary(self,index_1,index_2):
       #求a2 new 的取值范围 (L,H)
       a1_old=self.a[index_1][0]
       a2_old=self.a[index_2][0]
       if self.Y_train[index_1][0]!=self.Y_train[index_2][0]:
           # y1不等于y2的情况
           L = max(0,(a2_old-a1_old))
           H=min(self.C,(self.C+a2_old-a1_old))
       else:
           #y1 等于 y2的情况
           L=max(0,(a2_old+a1_old-self.C))
           H=min(self.C,(a2_old+a1_old))
       return L,H

   def cal_update_a1_a2(self,index_1,index_2):
       #计算更新后的a1，a2
       x1=self.X_train[index_1]
       x2=self.X_train[index_2]
       y1=self.Y_train[index_1][0]
       y2 =self.Y_train[index_2][0]
       a1_old = self.a[index_1][0]
       a2_old = self.a[index_2][0]
       E1=self.Ei[index_1]
       E2=self.Ei[index_2]
       k=self.select_kernel(x1,x1)+self.select_kernel(x2,x2)-2*self.select_kernel(x1,x2)
       a2_new = a2_old + y2*(E1-E2)/k
       L,H=self.cal_boundary(index_1,index_2)
       if a2_new>H:
           a2_new=H
       elif a2_new <L:
           a2_new=L
       a1_new=a1_old+y1*y2*(a2_old-a2_new)
       return a1_new,a2_new,a1_old,a2_old

   def update_b_Ei(self,index_1,index_2,a1_old,a2_old):
       #在每次完成两个变量的优化后，都要重新计算一下b
       x1 = self.X_train[index_1]
       x2 = self.X_train[index_2]
       y1 = self.Y_train[index_1][0]
       y2 = self.Y_train[index_2][0]
       a1_new= self.a[index_1][0]
       a2_new= self.a[index_2][0]
       E1 = self.Ei[index_1]
       E2 = self.Ei[index_2]
       b1_new=  -E1 - y1 * self.select_kernel(x1,x1) *  (a1_new - a1_old) - y2 * self.select_kernel(x2,x1) * (a2_new-a2_old) + self.b
       b2_new = -E2 - y1 * self.select_kernel(x1, x2) * (a1_new - a1_old) - y2 * self.select_kernel(x2, x2) * (a2_new - a2_old) + self.b
       if a1_new>0 and a1_new<self.C:
           self.b=b1_new
       elif a2_new>0 and a2_new<self.C:
           self.b=b2_new
       else:
           self.b=(b1_new+b2_new)/2
       # 支持向量机集合
      # data1 = self.a[(self.a[:, 0] > 0 and self.a[:, 0] < self.C), :]
       sum1=0
       sum2=0
       g1=self.cal_gx(x1)
       g2 = self.cal_gx(x2)
       self.Ei[index_1]=g1-y1
       self.Ei[index_2] =g2 - y2

   def smo(self):
      #smo算法
      select_variable,index_1,index_2=self.select_variable()
      i=0
      while select_variable: #如果仍有变量没有满足kkt条件
          print(i)
          i+=1
          a1_new,a2_new,a1_old,a2_old=self.cal_update_a1_a2(index_1,index_2)
          self.a[index_1][0] = a1_new#更新其值
          self.a[index_2][0] = a2_new
          self.update_b_Ei(index_1,index_2,a1_old,a2_old)
          print(self.predict())
          select_variable, index_1, index_2 = self.select_variable()


   def predict(self):
       right=0
       #data1 = self.a[(self.a[:, 0] > 0 and self.a[:, 0] < self.C), :]
       for i in range(self.X_test.shape[0]):
           f=0
           f=self.cal_gx(self.X_test[i])
           if f>0:
               y=1
           else:
               y=-1
           if y==self.Y_test[i][0]:
               right+=1
       return right/self.X_test.shape[0]


breast_cancer = datasets.load_breast_cancer()
data1 = breast_cancer.data
data2 = breast_cancer.target
data2.resize([data2.shape[0], 1])
for i in range(data2.shape[0]):
    if data2[i][0] == 0:
        data2[i][0] = -1
train_size = int(len(data1) * 0.7)  # 划分训练集与测试集
x_train = np.array(data1[:train_size])
y_train= np.array(data2[:train_size])
# print(X_train.shape)
x_test = np.array(data1[train_size:])
y_test= np.array(data2[train_size:])
#(self,X_train,Y_train,X_test,Y_test,kernel,d,sigma,C,e):
model = SVM(x_train, y_train, x_test,y_test,'rbf',2,25,1,0)
model.smo()
s=np.array([[1,2,3]])
p=np.array([[4,5,6]])
q= np.concatenate((s, p), axis=0)
print(q)