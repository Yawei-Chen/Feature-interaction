# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:35:58 2023

@author: Administrator
"""

import pandas as pd
from sklearn import metrics
import numpy as np
import math
from entroy import Entropy, jointEntropy

data1 = pd.read_csv(r'G:\动脉硬化\程序\AD_features\Extractrd_Features.csv')
data2 = pd.read_excel("G:\动脉硬化\数据\动脉硬化数据统计.xlsx") 
pwv = data2['PWV']

## 计算特征与AS之间的相关性和NMI,本论文算法1，1-6行
S=[]
R=[]
for i in range(data1.shape[1]):
    a=data1.iloc[:, i]
    s=metrics.normalized_mutual_info_score(a, pwv)
    r = a.corr(pwv)
    S.append(s)
    R.append(r)
   
    
######################
### 1  Rd: adjusted relevance measure 
### 2  F: selected features 
### 3  F_fe: all features
######################
F_fe = data1
K = 5
k = 0
w = [1 for _ in range(data1.shape[1])] 

F = []
while k<K:
    Rd=[] 
    for i in range(F_fe.shape[1]):       
        rd = w[i]*(1+metrics.normalized_mutual_info_score(F_fe.iloc[:, i], pwv))
        Rd.append(rd)
    
    F_j = max(Rd) # 求列表最大值
    F_jd = Rd.index(F_j) # 求最大值对应索引 
    
    #if (F_j>0):
    F_save = F_fe.iloc[:,F_jd].name
        
    F.append(F_save)
    F_fe = F_fe.drop(columns=F_fe.columns[F_jd])    #删除已选特征
    
   # if(F_j<0):
       #  break
    
    for j in range(F_fe.shape[1]):
        
        ###  实际上, I((f_i,f_j);T)的计算十分复杂，但是根据参考文献[1],
        ###  I(T;f_i;f_j) = I(f_i;f_j|T) - I(f_i;f_j)
        
        ### 近似计算有：I(f_i;f_j|T) = (I(f_i;f_j)/H(f_j))*H(f_j|T)
        
        ### [1]张俐,王枞.基于最大相关最小冗余联合互信息的多标签特征选择算法[J].通信学报,2018,39(05):111-122.
       
        ####################################
        # 1: fa, 上一步的已选特征
        # 2：fb, 待选特征集中的待选特征
        #####################################
        fa = data1[F_save]
        fb = F_fe.iloc[:, j]
        
        Hfa = Entropy(fa)
        Hfb = Entropy(fb)
        Hfa_fb = jointEntropy(fa,fb)
        Ifafb = Hfa-Hfa_fb
        
       
        I_thr = (Ifafb/Hfb)*jointEntropy(fb,pwv) - Ifafb       
        
        CC = I_thr/(Hfa + Hfb)
        
        w [j] = w [j]*CC
        
    k = k+1     
        

