#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : simple_logistic.py
# @Author: 刘绪光
# @Date  : 2018/5/26
# @Desc  : 简单的逻辑回归

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.001  # 学习率
iteration = 5000  # 迭代次数


def load_data():
    # 读入数据
    df = pd.read_csv('test.csv')

    # 取label标签
    Y_train = np.mat(df['class'])

    # 将行向量转换为列向量
    Y_train = np.mat(Y_train)
    Y_train = Y_train.T

    # 删除最后一列，即删除标签列
    df.drop('class', axis=1, inplace=True)

    # 添加一列，当b吧，方便计算，初始化为1
    df['new'] = 1
    X_train = np.mat(df)
    return X_train, Y_train


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradient_descent(X_train, Y_train):
    row, col = X_train.shape

    # 初始化，全为0
    W = np.zeros((col, 1))

    # 进行max_iteration次梯度下降
    for i in range(iteration):
        # 直接使用numpy提供的tanh函数
        h = np.tanh(np.dot(X_train, W))
        error = Y_train + h

        # 梯度下降
        W = W - alpha * np.dot(X_train.T, error)
    return W.getA()


# 这段代码来抄自https://github.com/apachecn/MachineLearning/blob/master/src/py2.x/ML/5.Logistic/logistic.pyu
def plot_show(W):
    X_train, Y_train = load_data()
    xcord1 = []
    ycord1 = []

    xcord2 = []
    ycord2 = []

    for i in range(X_train.shape[0]):
        if int(Y_train[i]) == 1:
            xcord1.append(X_train[i, 0])
            ycord1.append(X_train[i, 1])
        else:
            xcord2.append(X_train[i, 0])
            ycord2.append(X_train[i, 1])

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    x = np.arange(-4.0, 5.0, 0.1)
    """
    函数原型是：f(x) = w0*x0 + w1*x1 + b
    x1在这里被当做y值了，f(x)被算到w0、w1、b身上去了
    所以有：w0*x0 + w1*x1 + b = 0
    可以得到：(b + w0 * x) / -w1
    """
    y = (W[2] + W[0] * x) / -W[1]

    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.show()


if __name__ == '__main__':
    X_train, Y_train = load_data()
    # print(Y_train)
    W = gradient_descent(X_train, Y_train)
    print(W)
    plot_show(W)
