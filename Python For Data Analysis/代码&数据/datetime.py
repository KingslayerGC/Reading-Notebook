#!/usr/bin/env python
# coding: utf-8

# In[2]:


## 获取当前时间
from datetime import datetime
now = datetime.now()
now


# In[3]:


## 时间和时间差计算
from datetime import timedelta
start = datetime(2011, 1, 7)
print("after 12 days, here comes", start + 2* timedelta(12))

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
print("there is a gap about", delta)


# In[4]:


## 字符串和datetime对象的相互转换

stamp = datetime(2011, 1, 3)
print(str(stamp))
print(stamp.strftime('%Y - %m - %d'))

# 自定义解析字符串
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')
# 使用parse方法(自动识别模式)
from dateutil.parser import parse
parse('2011,01,03')
parse('Jan 31, 1997 10:45 pm')
parse('6/12/2011', dayfirst=True)


# In[5]:


## pandas处理成组日期
import pandas as pd
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00', None]
pd.to_datetime(datestrs)


# In[17]:


## 日期索引的生成和引用

pd.date_range(start='1/1/2000', periods=100, freq='W-WED')
pd.date_range(start='2000-01-01', end='2000-12-01', freq='BM')
pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4h30min')
pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI')

import numpy as np
longer_ts = pd.Series(np.random.randn(1000),
                      index=pd.date_range('1/1/2000', periods=1000))
longer_ts['2001-01']
longer_ts['9/20/2002':'1/1/2003']


# In[41]:


## 沿时间轴移动数据或时间
ts = pd.Series(np.random.randn(4),
               index=pd.date_range('1/1/2000', periods=4, freq='M'))

# 移动数据
ts / ts.shift(1) - 1 #计算增长率

# 移动时间索引
ts.shift(1, freq='30T') #向后移动30min

# 移动datetime或Timestamp对象
from pandas.tseries.offsets import Day, MonthEnd
now = datetime.now()
now - Day(3)
now + MonthEnd(2) # 接下来的第二个月末


# In[40]:


## 时区操作

# 创建有时区信息的时间戳
stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
print(stamp_moscow.value) #UTC时间，不随时区转换而变化

# 时间序列本地化到某个特定时区
rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts_utc = ts.tz_localize('UTC')
print(ts_utc.index.tz)

# 时间序列转换时区
ts_utc.tz_convert('America/New_York') #由于夏令时，时间变动不一定一致

# 不同时区的序列运算
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
print(result.index.tz) #直接输出UTC结果


# In[64]:


## 时期
### 相比datetime对象，period对象更像⼀个时间段中的游标

# 时期的计算
p = pd.Period(2007, freq='A-DEC')
p - 2
pd.Period('2014', freq='A-DEC') - p

# 时期序列
rng = pd.period_range('2000-01-01', '2000-06-30', freq='M')
pd.Series(np.random.randn(6), index=rng)

# 字符串转时期
values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')

# 时期的频率转换
p.asfreq('M', how='start')
rng = pd.period_range('2006', '2009', freq='A-DEC')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.asfreq('M', how='start'), '\n', ts.asfreq('B', how='end'))


# In[69]:


## 时间戳与period的转换

rng = pd.date_range('1/29/2000', periods=6, freq='D')
ts2 = pd.Series(np.random.randn(6), index=rng)
pts = ts2.to_period('M')
print(pts)

pts.to_timestamp(how='end')


# In[94]:


## 重采样

# 降采样
rng = pd.date_range('2000-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.resample('M', kind='period').mean()

rng = pd.date_range('2000-01-01', periods=12, freq='T')
ts = pd.Series(np.arange(12), index=rng)
ts.resample('5min', closed='right', label='right', loffset='-1s').sum()

# OHLC重采样
ts.resample('5min', closed='right').ohlc()

# 升采样
frame = pd.DataFrame(np.random.randn(2, 4),
                     index=pd.date_range('1/1/2000', periods=2,freq='W-WED'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame.resample('D').asfreq() #不填充
frame.resample('D').ffill(limit=2) #限制填充次数

# 时期索引重采样
frame = pd.DataFrame(np.random.randn(24, 4),
                     index=pd.period_range('1-2000', '12-2001',freq='M'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])
annual_frame = frame.resample('Y').mean()
annual_frame.resample('Q-DEC').ffill()


# In[146]:


## 窗口函数
import matplotlib.pyplot as plt
close_px_all = pd.read_csv(r"C:\Users\Mac\Desktop\过程\课外\电子书\Python For Data Analysis\代码&数据\stock_px_2.csv",
                           parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']].resample('B').ffill()
fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax = close_px.AAPL.plot(label="daily price")
ax = close_px.AAPL.rolling(250, min_periods=10).mean().plot(label="250 days mean price")
ax = close_px.AAPL.expanding(250).mean().plot(label="cumsum mean price")
ax.tick_params(labelsize=20)
ax.legend(loc='best', fontsize=20)
ax.set_title("Stock Price Analysis", fontsize=25)


# In[181]:


# 不规则滚动
close_px.rolling('20D').mean()

# 指数加权移动平均
aapl_px = close_px.AAPL['2006':'2007']
aapl_px.ewm(span=30).mean()

# 滚动相关系数
spx_px = close_px_all['SPX']
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()
returns.rolling(125, min_periods=100).corr(spx_rets)

# 自定义：滚动五分位数
from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent)


# In[ ]:




