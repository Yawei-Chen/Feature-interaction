# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:39:13 2023

@author: Administrator
"""
import pandas as pd
import numpy as np
import math
from collections import Counter






# 定义计算信息熵的函数
def Entropy(DataList):
    '''
        计算随机变量的熵
    '''
    counts = len(DataList)      # 总数量
    counter = Counter(DataList) # 每个变量出现的次数
    prob = {i[0]:i[1]/counts for i in counter.items()}      # 计算每个变量的 p*log(p)
    H = - sum([i[1]*math.log2(i[1]) for i in prob.items()]) # 计算熵
    
    return H


def jointEntropy(X,Y):
    
    XY = list(zip(X,Y))
    HX = Entropy(X)   # 随机变量 X 的熵
    HY = Entropy(Y)   # 随机变量 Y 的熵
    HXY = Entropy(XY) # 联合熵 XY
    HX_Y = HXY - HY   # 条件熵  X｜Y
    
    
    return HX_Y



# 定义计算信息熵的函数：计算Infor(D)
def infor(data):
    a = pd.value_counts(data) / len(data)     #pd.value_counts()每个数字或字符的个数, a是每个字符个数除以长度
    return sum(np.log2(a) * a * (-1))
