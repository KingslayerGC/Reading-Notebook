#!/usr/bin/env python
# coding: utf-8

# In[26]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'


# In[4]:


import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
### statsmodels的线性模型有两种不同的接⼝：基于数组和基于公式。它们可以通过API模块引⼊。


# In[12]:


## 定义正态随机数生成函数
def dnorm(mean, variance, size=1, random_state=42):
    if isinstance(size, int):
        size=(size,)
    return mean + np.sqrt(variance) * np.random.randn(*size)


# In[50]:


## 拟合一个最小二乘回归
# 生成一组数据
N = 100
X = np.c_[dnorm(0, 0.4, N), dnorm(0, 0.6, N), dnorm(0, 0.2, N)]
eps = dnorm(50, 0.1, size=N)
beta = [10, 20, 30]
y = np.dot(X, beta) + eps

# 构造模型方法一
X_c = sm.add_constant(X)
model = sm.OLS(y, X_c)
result = model.fit()

# 构造模型方法二
data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
result = smf.ols('y ~col0 + col1 + col2', data=data).fit() #参考patsy字符串方法

# 结果查看
print(result.summary())
result.params
result.tvalues
result.predict(data.loc[0:5])


# In[ ]:




