#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : knn.py
# @Author: 刘绪光
# @Date  : 2018/6/5
# @Desc  :

import pandas as pd
import numpy as np
import operator


def get_data():
    """
    读取数据
    :return: 数据集和标签集
    """
    df = pd.read_csv('hailun.csv')

    labels = np.array(df['label'])

    df.drop('label', axis=1, inplace=True)
    data = np.mat(df)

    return data, labels


def normalization(data):
    """
    归一化处理，归一化公式为：Y = (X-Xmin)/(Xmax-Xmin)
    :param data: 待处理数据
    :return: 处理后的数据
    """
    min_val = data.min(0)
    max_val = data.max(0)

    # 极差
    ranges = max_val - min_val

    norm_data = (data - min_val) / ranges

    return norm_data, ranges, min_val


def classify(inX, norm_data, labels, k):
    """
    :param inX: 待分类数据
    :param norm_data: 输入的训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return: 分类标签
    """
    # 这里使用的是欧式距离
    data_size = norm_data.shape[0]
    diff_mat = np.tile(inX, (data_size, 1)) - norm_data
    sq_diff_mat = np.array(diff_mat) ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = np.array(sq_distances) ** 0.5

    # 将距离排序：从小到大
    sorted_dist_indicies = distances.argsort()

    # 选择前k个，并选择最多的标签类别
    clazz_count = {}
    for i in range(k):
        label = labels[sorted_dist_indicies[i]]
        clazz_count[label] = clazz_count.get(label, 0) + 1

    sorted_clazz = sorted(clazz_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_clazz[0][0]


if __name__ == '__main__':
    data, labels = get_data()
    norm_data, ranges, min_val = normalization(data)
    inX = np.array([26052, 1.441871, 0.805124])
    a = classify((inX - min_val)/ranges, norm_data, labels, 5)
    print(a)
