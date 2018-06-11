#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : NaiveBayes.py
# @Author: 刘绪光
# @Date  : 2018/6/11
# @Desc  :
import re
import collections


class SpellCheck():
    """
    P(c|w) = P(w|c)P(c) / P(w)
    P(c)：文章出现正确拼写词c的概率，程序中直接用词频表示
    P(w|c)：用户把词c错敲成w的概率
    """
    alphabet = list('abcdefghijklmnopqrstuvwxyz')

    # 读取数据
    def get_data(self):
        fp = open('big.txt', 'r', encoding='utf-8')
        return fp.read()

    # 只拿文本中的单词
    def get_words(self):
        text = self.get_data().lower()
        return re.findall('[a-z]+', text)

    # 词频统计，返回的是词语和出现的次数
    def train(self):
        result = collections.defaultdict(lambda: 1)
        for word in self.get_words():
            result[word] += 1
        return result

    def edit_first(self, word):
        """
        只编辑一次就把一个单词变为另一个单词
        :return: 所有与单词word编辑距离为1的集合
        """
        length = len(word)
        return set([word[0:i] + word[i + 1:] for i in range(length)] +  # 删除一个字母
                   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(length - 1)] +  # 调换两个字母
                   [word[0:i] + c + word[i + 1:] for i in range(length) for c in self.alphabet] +  # 修改一个字母
                   [word[0:i] + c + word[i:] for i in range(length + 1) for c in self.alphabet])  # 插入一个字母

    def edit_second(self, word):
        """
        :return: 编辑两次的集合
        """
        words = self.train()
        return set(e2 for e1 in self.edit_first(word) for e2 in self.edit_first(e1) if e2 in words)

    def already_words(self, word):
        """
        :return: 返回已知的和错误单词相近的正确单词集合，允许进行两次编辑
        """
        words = self.train()
        return set(w for w in word if w in words)

    def check(self, word):
        words = self.train()
        neighborhood = self.already_words([word]) \
                       or self.already_words(self.edit_first(word)) \
                       or self.already_words(self.edit_second(word)) \
                       or [word]
        # 取概率最大的正确单词，即词频最多的
        return max(neighborhood, key=lambda w: words[w])


if __name__ == '__main__':
    s = SpellCheck()
    print(s.check('learb'))
