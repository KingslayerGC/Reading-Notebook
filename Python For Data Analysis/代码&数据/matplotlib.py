#!/usr/bin/env python
# coding: utf-8

# In[4]:


## 开启交互式绘图
get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


## 创建子图方法一
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
ax3 = plt.plot(np.random.randn(50).cumsum(), 'k--')


# In[15]:


## 创建子图方法二
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)


# In[16]:


## 采用不同的插值方法
import numpy as np
fig = plt.figure()
data = np.random.randn(30).cumsum()
plt.plot(data, 'k--', label='Default')
plt.plot(data, 'k-',  drawstyle='steps-post', label='Steps-post')
plt.legend(loc='best')


# In[17]:


## 设置坐标轴刻度和刻度标签
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')
ax.legend(["random walk"])
plt.subplots_adjust(top=0.85, bottom=0.15)


# In[19]:


## 设置注解、箭头图标
from datetime import datetime

data = pd.read_csv(r"C:\Users\Mac\Desktop\过程\课外\电子书\Python For Data Analysis\代码&数据\spx.csv", index_col=0, parse_dates=True)
spx = data['SPX']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(data, 'k-')
crisis_data = [
(datetime(2007, 10, 11), 'Peak of bull market'),
(datetime(2008, 3, 12), 'Bear Stearns Fails'),
(datetime(2008, 9, 15), 'Lehman Bankruptcy')
]
for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                horizontalalignment='left', verticalalignment='top')
# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])
ax.set_title('Important dates in the 2008-2009 financial crisis')


# In[20]:


## 绘制几何图形
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color='g', alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
#plt.savefig(r"C:\Users\Mac\Desktop\usersfigpath.png", dpi=400, bbox_inches='tight')


# In[21]:


## 使用pandas内置线形图
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df.plot()


# In[71]:


## 使用series内置柱形图
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot.bar(ax=axes[0], color='k', alpha=0.7)
axes[0].set_xticklabels(labels=data.index, rotation=0)
data.plot.barh(ax=axes[1], color='k', alpha=0.7)
axes[1].set_yticklabels(labels=[data.index[i] if i%2==0 else None for i in range(len(data.index))], rotation=0)


# In[23]:


## 使用pandas内置柱形图一
df = pd.DataFrame(np.random.rand(6, 4),
                  index=['one', 'two', 'three', 'four', 'five', 'six'],
                  columns=pd.Index(['A', 'B', 'C', 'D'],name='Genus'))
ax = df.plot.barh(stacked=True, alpha=0.5)
ax.set_xticklabels(labels=ax.get_xticks(), rotation=30)
ax.set_yticklabels(labels=df.index, rotation=30)


# In[5]:


## 使用pandas内置柱形图二
tips = pd.read_csv(r"C:\Users\Mac\Desktop\过程\课外\电子书\Python For Data Analysis\代码&数据\tips.csv", header=0)
party_counts = pd.crosstab(tips['day'], tips['size'])
party_pcts = party_counts.div(party_counts.sum(1), axis=0)
party_pcts.plot.bar()


# In[84]:


## 直⽅图和连续密度估计图
tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
fig, axes = plt.subplots(3, 1, figsize=(6, 7))
tips['tip_pct'].plot.hist(ax=axes[0], bins=50)
axes[0].set_ylabel(None)
tips['tip_pct'].plot.density(ax=axes[1])
axes[1].set_ylabel(None)
comp1 = np.random.normal(0, 1, size=200)
comp2 = np.random.normal(10, 2, size=200)
values = pd.Series(np.concatenate([comp1, comp2]))
sns.distplot(values, bins=100, color='k', ax=axes[2])
fig.subplots_adjust(top=0.95, bottom=0.075, wspace=0.3, hspace=0.2)


# In[67]:


## 使用seaborn绘制散点图一
plt.close('all')
macro = pd.read_csv(r"C:\Users\Mac\Desktop\过程\课外\电子书\Python For Data Analysis\代码&数据\macrodata.csv")
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
sns.regplot('m1', 'unemp', data=trans_data)
plt.title('Changes in log %s versus log %s' % ('m1', 'unemp'))


# In[64]:


## 使用seaborn绘制散点图二
plt.close('all')
sns.pairplot(trans_data, diag_kind='kde', plot_kws={'alpha': 0.2})


# In[68]:


## 使用seaborn对合计数据绘制直方图一
plt.close('all')
sns.factorplot(x='day', y='tip_pct', hue='time', col='smoker',kind='bar', data=tips[tips.tip_pct < 1])
#　设置风格，默认为'darkgrid'
sns.set(style="whitegrid")
# 擦除上侧和右侧轴脊柱
sns.despine()


# In[69]:


## 使用seaborn对合计数据绘制直方图二
sns.factorplot(x='day', y='tip_pct', row='time', col='smoker', kind='bar', data=tips[tips.tip_pct < 1])


# In[126]:


## 使用seaborn绘制箱型图
plt.close('all')
plt.figure(figsize=(7, 7))
sns.boxplot(x='tip_pct', y='day', data=tips[tips.tip_pct < 0.5])


# In[ ]:




