

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:01:14 2023

@author: Administrator
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.model_selection import cross_val_score, cross_val_predict

import xgboost as xgb
from xgboost import plot_importance


# data = pandas.read_csv('Features.csv')
# X = data[['K','T_up','T_tol','slope','eata','re_hight','mid_hight']]

data1 = pd.read_csv(r'G:\动脉硬化\程序\AD_features\Extractrd_Features.csv')
data2 = pd.read_excel("G:\动脉硬化\数据\动脉硬化数据统计.xlsx") 
pwv = data2['PWV']

X = data1[['SI','PNL','CL','PNH','AI','K','VFD','Hda']]
#X = data1.iloc[:, 0:58]
X=np.array(X)



'''
# 仅用Box变换，mae = 0.2951
converted_x1 = stats.boxcox(X[:,0])[0]
converted_x2 = stats.boxcox(X[:,1])[0]
converted_x3 = stats.boxcox(X[:,2])[0]
converted_x4 = stats.boxcox(X[:,3])[0]
converted_x5 = stats.boxcox(X[:,4])[0]
converted_X = np.c_[converted_x1,converted_x2,converted_x3,converted_x4,converted_x5]
# sns.distplot(converted_x1)
'''

Y = pwv
Y=np.array(Y)
Y=Y.reshape((196,1))
# sns.distplot(Y)

#对标签进行BOX-COX变换
# converted_Y = stats.boxcox(Y)[0] 
# sns.distplot(converted_data1)



# 数据集制作
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=6,8,9,11,16,17);  # 默认3：1划分
# ('K','slope','time','tol_s','ER1','ER2','ER3',)  random_state=3,8,9,10,11,12,16,17,18,
# 'K','slope','time','tol_s','ER1','ER2','ER3','fr_HR','fr_max'， 2，3，8，9，11，12，16
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=7);  # 默认3：1划分


# xgboost

params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

dtrain = xgb.DMatrix(X_train, Y_train)
num_rounds = 300
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

# 预测
dtest = xgb.DMatrix(X_test)
Y_predit = model.predict(dtest)

Y_predit=Y_predit.reshape((Y_predit.shape[0],1))


plot_importance(model)
plt.show()



# 决定系数评价指标  
from sklearn.metrics import r2_score
R2 =  r2_score(Y_test, Y_predit)
print('确定系数R2',R2)


fig=plt.figure()
plt.plot(np.arange(len(Y_test)), Y_test,'go-',label='reference TG')
plt.plot(np.arange(len(Y_predit)),Y_predit,'ro-',label='predicted TG')
plt.xlabel('sample')
plt.ylabel('TG(mmol/L)')
plt.legend()
plt.show()

# calculate error\n",
mae = mean_absolute_error(Y_test, Y_predit)
std = np.std(np.abs(Y_test-Y_predit))
PCC=stats.pearsonr(Y_test[:,0],Y_predit[:,0])[0]
print('MAE:', mae)
print('STD:', std)
print('PCC:', PCC)