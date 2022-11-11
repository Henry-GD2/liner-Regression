# liner-Regression
线性回归
#multiple-liner_regression.py
是用来查看各项参数和结果的线性相关性，以挑选出比较好的参数来进行训练
mod2 = smf.ols(formula='medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+b+lstat',data=bostondf)
这个代表每个参数都是一次的方程，如果需要多元多次(bostondf['crim']=bostondf['crim']^2)就变成二次了
res2 = mod2.fit()
print(res2.summary()
这句是查看结果

#training.py
正式开始训练，先把参数全部选好，摆放整齐，分为测试集和训练集
查看结果，画图
