import numpy as np
import random
class PLSA:
    def __init__(self,X,K,iterations):
        # 待潜在分析的 单词-文本矩阵
        self.X=X
        # 有K个话题
        self.K=K
        # EM算法迭代次数
        self.iterations=iterations
        #单词数
        self.M=self.X.shape[0]
        #文本数
        self.N=self.X.shape[1]
        # 在话题为z的条件下 单词为w的概率矩阵
        self.p_w_z=self.initialize_p_w_z()
        #在文本为d的条件下 话题为 z的概率矩阵
        self.p_z_d=self.initialize_p_z_d()
        # 在单词为w ，文本为d 的条件下 话题为z的条件概率
        self.p_z_w_d=[[[0.0]*self.K]*self.M]*self.N

    def initialize_p_w_z(self):
        #初始化在某个话题条件下某个单词的条件概率
        #P为一个M行 K列矩阵 元素P[i][j]代表当话题为j时 单词为i的概率
        #以下为初始化元素为0到1之间的小数
        P=[[0.0]*self.K]*self.M
        for i in range(self.M):
            for j in range(self.K):
                P[i][j]=random.random()
        P_array=np.array(P)
        return P_array

    def initialize_p_z_d(self):
        # 初始化在已知文本为文本d条件下话题为z的条件概率
        # P为一个K行 N列矩阵 元素P[i][j]代表当文本为j时 话题为i的概率
        # 以下为初始化元素为0到1之间的小数
        P = [[0.0] * self.N] * self.K
        for i in range(self.K):
            for j in range(self.N):
                P[i][j] = random.random()
        P_array = np.array(P)
        return P_array

    def cal_E(self):
        #EM算法的E步 求《统计学习方法》公式18.11
        #计算在单词为w 文本为d的条件下 话题为z的条件概率
        for i in range(self.N): #遍历文本
            for j in range(self.M):# 遍历单词
                for k in range(self.K): #遍历话题
                    p1=self.p_w_z[j][k]*self.p_z_d[k][i]
                    p2=0
                    for k1 in range(self.K):
                        p2+=self.p_w_z[j][k1]*self.p_z_d[k1][i]
                    self.p_z_w_d[i][j][k]=p1/p2

    def cal_M(self):
        #EM算法M步
        # 求《统计学习方法》公式18.12
        for i in range(self.M): #遍历单词
            for k in range(self.K): #遍历话题
                p1=0
                for j in range(self.N):# 遍历文本
                    p1+=self.X[i][j]*self.p_z_w_d[j][i][k]
                p2=0
                for m in range(self.M):
                    for j in range(self.N):
                        p2+=self.X[m][j]*self.p_z_w_d[j][m][k]
                self.p_w_z[i][k]=p1/p2
        # 求《统计学习方法》公式18.13
        # 单词文本矩阵每一列相加 得到的是每一个文本中有多少个单词
        p2=self.X.sum(axis=0)
        for k in range(self.K):
            for j in range(self.N):
                p1=0
                for i in range(self.M):
                    p1+=self.X[i][j]*self.p_z_w_d[j][i][k]
                self.p_z_d[k][j]=p1/p2[j]

    def fit(self):
        for i in range(self.iterations):
            self.cal_E()
            self.cal_M()

X = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1],
              [1, 0, 0, 0, 0, 1, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 2, 0, 0, 1],
              [1, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 1, 1, 0, 0, 0, 0]])

plsa=PLSA(X,3,1000)
plsa.fit()
print(plsa.p_w_z)
print(plsa.p_z_d)