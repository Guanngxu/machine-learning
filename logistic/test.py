#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: 刘绪光
# @Date  : 2018/5/26
# @Desc  :

import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('test.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


if __name__ == '__main__':
     a,b = loadDataSet()
     # [-1.7612000e-02  1.4053064e+01  0.0000000e+00]
     # [1.0000000e+00 - 1.7612000e-02  1.4053064e+01]
     print(np.mat(a).shape)
     print(np.mat(b).transpose().shape)
