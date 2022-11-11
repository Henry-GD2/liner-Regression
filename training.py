# -*- coding: utf-8
#  @Time    : 2021/1/26 16:16
#  @Author  : ZYX
#  @File    : Model1_波士顿房价线性回归.py
# @software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1.获取数据集
boston_data = load_boston()
x = pd.DataFrame(boston_data.data) # 波士顿房价data
y = boston_data.target             # 波士顿房价真实值
x.columns = boston_data.feature_names # 特征赋值

# 2.划分训练集、测试集
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=125)
# 3.建立线性回归模型
reg = LinearRegression().fit(xtrain,ytrain)
# 4.1 获取预测值
y_pred = reg.predict(xtest)
# 4.2 获取回归系数
y_w = reg.coef_
# [-1.14077285e-01  4.87165173e-02 -1.20875379e-02  1.59355488e+00, -1.89792822e+01  3.47313008e+00  3.03276293e-03 -1.60090878e+00,  2.90563127e-01 -1.27238844e-02 -9.76743908e-01  8.48566379e-03, -4.87508387e-01]
# 4.3 获取截距
y_w0 = reg.intercept_ # 40.44599864104647
# 4.4 将回归系数与特征对应
compare_feature = [*zip(xtrain.columns,y_w)]
# [('CRIM', -0.11407728518733692),
# ('ZN', 0.04871651733969727),
# ('INDUS', -0.012087537917785052),
# ('CHAS', 1.5935548762536438),
# ('NOX', -18.97928220894716),
# ('RM', 3.4731300808153214),
# ('AGE', 0.0030327629267113626),
# ('DIS', -1.6009087800054767),
# ('RAD', 0.29056312669900103),
# ('TAX', -0.01272388444870162),
# ('PTRATIO', -0.976743908479199),
# ('B', 0.008485663789765877),
# ('LSTAT', -0.48750838710289024)]


# 5.预测结果可视化
plt.rcParams['font.sans-serif'] = 'SimHei'
fig = plt.figure(figsize=(10,6))
plt.plot(range(ytest.shape[0]),ytest,color='black',linestyle='-',linewidth=1.5)
plt.plot(range(y_pred.shape[0]),y_pred,color='red',linestyle='-.',linewidth=1.5)
plt.xlim((0,102))
plt.ylim((0,55))
plt.legend(['真实值','预测值'])
plt.show()



# 6.评价回归模型
from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,median_absolute_error,r2_score
# 平均绝对误差
mean_absolute_score = mean_absolute_error(y_pred=y_pred,y_true=ytest) # 3.3775517360082032
# 均方误差
mean_squared_score = mean_squared_error(y_pred=y_pred,y_true=ytest) # 31.15051739031563
# 中值绝对误差
median_absolute_score = median_absolute_error(y_pred=y_pred,y_true=ytest) # 1.7788996425420773
# 可解释方差
explained_variance_score = explained_variance_score(y_pred=y_pred,y_true=ytest) # 0.710547565009666
# R2
r2_score = r2_score(y_pred=y_pred,y_true=ytest) # 0.7068961686076838
