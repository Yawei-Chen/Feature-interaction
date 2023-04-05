# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:17:47 2023

@author: Administrator
"""

import numpy as np

def complementary(true_labels, predicted_labels, target):
    """
    计算互补系数

    :param true_labels: 真实的类别标签
    :param predicted_labels: 预测的类别标签
    :return: 归一化互信息
    """
    # 计算真实类别标签和预测类别标签的熵
    H_true = entropy(true_labels)
    H_predicted = entropy(predicted_labels)

    # 计算联合熵
    joint_entropy = entropy(np.column_stack((true_labels, predicted_labels)))

    # 计算互信息
    MI = H_true + H_predicted - joint_entropy

    # 计算归一化互信息
    NMI = 2 * MI / (H_true + H_predicted)

    return NMI

def entropy(labels):
    """
    计算信息熵

    :param labels: 类别标签
    :return: 熵
    """
    # 计算类别频率
    counts = np.bincount(labels)
    freq = counts / len(labels)

    # 计算熵
    H = -np.sum(freq * np.log2(freq))
    
    return H
