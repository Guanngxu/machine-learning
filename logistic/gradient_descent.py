#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 逻辑回归入门.py
# @Author: 刘绪光
# @Date  : 2018/5/26
# @Desc  : 梯度下降计算函数的极小值

"""
f(x) = x^2 + 3x + 4
f(x)的导数  g(x) = 2x + 3
"""
def test():
    def derivative(x_pre):  # f(x)的导数
        return 2 * x_pre + 3

    x_pre = -5  # 初始值
    x_now = 1  # 梯度下降初始值
    alpha = 0.01  # 学习率，即步长
    pression = 0.00001  # 更新的阀值
    count = 0  # 统计迭代次数

    while abs(x_now - x_pre) > pression:
        x_pre = x_now
        # x = x - αg(x)，g(x)为f(x)的导数
        x_now = x_pre - alpha * derivative(x_pre)
        count += 1
    print(x_now)
    print(count)


if __name__ == '__main__':
    test()
