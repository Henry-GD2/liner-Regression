import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# 读取网络数据
boston = load_boston()
# 数据包含14个字段，boston.data是前13个字段数据，boston.target是第13个字段'medv'的数据
col = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']
bostondf = pd.DataFrame(boston.data,columns=col)
bostondf['medv']=boston.target
bostondf.head()

#bostondf是创建的DataFrame的类型，本示例给各个环节加上了栏目名字，并且把价格栏也加上了

import statsmodels.formula.api as smf

#print(bostondf['crim']*2)#此处可以平方，然后在下个式子里就成了曲线回归
mod2 = smf.ols(formula='medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+b+lstat',data=bostondf)
res2 = mod2.fit()
print(res2.summary())
#查看评估结果，其中决定系数R-squared为0.741，可以认为拟合得比较好，一般来说P值决定相关性，一般认为P值<0.05线性相关比较好，而相反线性相关比较差
#从结果可以看到,indus和age与结果线性相关不好
print(res2.params)#拟合出的结果
#结果的coef就是相关系数，而const就是偏置项（常数）

y_fitted = res2.fittedvalues
fig, ax = plt.subplots(figsize=(8,6))
print(np.array(bostondf['crim'][0]))
ax.plot(np.array(bostondf['crim']), np.array(bostondf['medv']), 'o', label='data')
ax.plot(np.array(bostondf['crim']), y_fitted, 'r--.',label='OLS')
ax.legend(loc='best')
plt.show()
#画图，这里只用了第一个系数crim显示结果

