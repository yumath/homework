import math
import random
from numpy import *
import numpy as np
import time
from threading import Thread
import functools

#求nca中的exp函数
def Exp(A, x, y):
    z = np.linalg.norm(A.dot(x) - A.dot(y))
    z = z**2
    return math.exp(-z)

def train(traindata):
    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    X = traindata[0]
    Y = traindata[1]
    #time.sleep(1) # 这行仅用于测试训练超时，运行时请删除这行，否则你的TRAINING_TIME_LIMIT将-1s。
    old_dimention = X.shape[1]    #原空间的维度
    n_points = X.shape[0]  # 待训练的元素个数

    #将度量矩阵设置为全局变量
    global A
    A = mat(eye(old_dimention, old_dimention, dtype=float))
    alpha = 0.1   #训练步长
    delta = 5.0     #收敛阈值
    F_last = 0      #上一个函数值

    #开始训练
    print("Start training!")
    while delta > 0.1:
        P = np.zeros(n_points)  #一维矩阵，存储每个点被正确分类的概率
        _gradient = mat(zeros((old_dimention, old_dimention))) #二维矩阵，梯度方向
        t = random.randint(0, n_points-10)  #设置一个随机样本起点
        for i in range(t, t+10, 1):
            #当前节点及其label
            cur_point = X[i]
            cur_label = Y[i]
            # 先求关于i的分母
            cur_denominator = 0
            for k in range(n_points):
                cur_denominator += Exp(A, cur_point, X[k])
            #初始化参数
            theta1 = theta2 = mat(zeros((old_dimention, old_dimention)))
            for j in range(n_points):
                p_ij = Exp(A, cur_point, X[j])/cur_denominator
                product_ij = p_ij*((mat(cur_point)-mat(X[j])).T)*(mat(cur_point)-mat(X[j]))
                theta2 += product_ij    # 求theta2
                if i != j and Y[j] == cur_label:
                    P[i] += p_ij   #求每个P[i]
                    #求theta1
                    theta1 += product_ij
            theta2 = P[i]*theta2
            _gradient += theta1 - theta2
        A = A - alpha * _gradient   #更新度量矩阵
        F = 1 - P.sum()
        delta = abs(F - F_last)     #计算两次函数值之间的差
        print("当前F值为：", F, "\t损失值为：", delta)
        F_last = F
    print("training is over!")

def distance(inst_a, inst_b):
    #dist = np.random.rand() # 随机度量 这是一个错误示范，请删除这行 _(:3 」∠)_
    dist = np.linalg.norm(A.dot(inst_a) - A.dot(inst_b))
    return dist
