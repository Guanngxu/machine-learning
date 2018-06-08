#!/usr/bin/python
# coding:utf-8

import operator
import math
import decisionTreePlot


def create_data():
    """
    创造测试数据
    :return: 测试数据
    """
    data_set = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    label_set = ['no surfacing', 'flippers']
    return data_set, label_set


def get_iris_data():
    """
    读取鸢尾花数据
    :return: 数据集和特征集
    """

    fp = open('iris.csv', 'r', encoding='utf-8')
    data_set = []

    for line in fp:
        data = line.strip().replace('\n', '').split(',')
        data_set.append(data)

    label_set = ['a', 'b', 'c', 'd']
    return data_set, label_set


def get_lenses_data():
    fp = open('lenses.csv', 'r', encoding='utf-8')
    data_set = []

    for line in fp:
        data = line.strip().replace('\n', '').split('	')
        data_set.append(data)

    label_set = ['age', 'prescript', 'astigmatic', 'tearRate']
    return data_set, label_set


def calc_entropy(data):
    """
    计算数据集的香农熵
    :param data:
    :return:
    """
    # 统计类别出现次数
    label_count = {}
    for iris in data:
        cur_label = iris[-1]
        if cur_label not in label_count.keys():
            label_count[cur_label] = 1
        label_count[cur_label] += 1

    shannon_ent = 0.0
    length = len(data)
    for key in label_count:
        prob = float(label_count[key]) / length
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def split_data(data, index, value):
    """
    划分数据集
    :param data: 待划分数据集
    :return:
    """
    ret_data_set = []
    for feat in data:
        if feat[index] == value:
            reduced_feat = feat[:index]

            reduced_feat.extend(feat[index + 1:])
            ret_data_set.append(reduced_feat)
    return ret_data_set


def choose_feat(data_set):
    """
    选择最优划分特征
    :param data: 数据集
    :return: 最优特征的索引
    """

    # 共有多少个特征，减一是因为最后一列为标签值
    length = len(data_set[0]) - 1
    base_ent = calc_entropy(data_set)

    # 最优信息增益值、最优特征索引
    best_info_gain, best_feat_index = 0.0, -1
    # iterate over all the features
    for i in range(length):
        # 这里使用了list生成式
        feat_list = [example[i] for example in data_set]

        # 去重
        unique_val_set = set(feat_list)
        cur_entropy = 0.0
        for value in unique_val_set:
            sub_data = split_data(data_set, i, value)
            prob = len(sub_data) / float(len(data_set))
            cur_entropy += prob * calc_entropy(sub_data)
        info_gain = base_ent - cur_entropy
        if (info_gain > base_ent):
            best_info_gain = info_gain
            best_feat_index = i
    return best_feat_index


def major_cnt(class_list):
    """
    选择出现次数最多的结果
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    构建决策树
    :param data_set: 数据集
    :param labels: 特征集集
    :return: 构建好的决策树
    """

    # 如果数据只有一个类别，那就直接返回
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 如果数据只有1列，那么出现label次数最多的一类作为结果
    if len(data_set[0]) == 1:
        return major_cnt(class_list)

    best_feat = choose_feat(data_set)
    # 获取label的名称
    best_feat_label = labels[best_feat]

    des_tree = {best_feat_label: {}}
    del (labels[best_feat])

    feat_vals = [example[best_feat] for example in data_set]
    unique_vals = set(feat_vals)
    for value in unique_vals:
        # 求出剩余的标签label
        sub_labels = labels[:]
        # 递归调用函数create_tree()，继续划分
        des_tree[best_feat_label][value] = create_tree(split_data(data_set, best_feat, value), sub_labels)
    return des_tree


def classify(tree, feat_labels, test):
    """
    :param tree: 决策树
    :param feat_labels: 特征集
    :param test: 测试数据
    :return: 预测结果
    """
    first = list(tree.keys())[0]

    # 通过key得到根节点对应的value
    second_dict = tree[first]
    feat_index = feat_labels.index(first)
    key = test[feat_index]
    val_of_feat = second_dict[key]
    if isinstance(val_of_feat, dict):
        res = classify(val_of_feat, feat_labels, test)
    else:
        res = val_of_feat
    return res


def test():
    # 1.创建数据和结果标签
    data_set, labels = get_iris_data()

    import copy
    des_tree = create_tree(data_set, copy.deepcopy(labels))
    print(des_tree)
    # print(classify(des_tree, labels, [1, 0]))
    decisionTreePlot.createPlot(des_tree)


if __name__ == "__main__":
    test()
    # get_lenses_data()
