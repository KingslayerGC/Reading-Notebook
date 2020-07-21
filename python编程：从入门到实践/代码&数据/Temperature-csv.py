#!/usr/bin/env python
# coding: utf-8

# In[27]:


import csv

#filename=r"C:\Users\Mac\Desktop\ehmatthes-pcc_2e-1.1-0-gbc387ba\ehmatthes-pcc_2e-bc387ba\chapter_16\the_csv_file_format\data\sitka_weather_07-2018_simple.csv"
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    for index, column_reader in enumerate(header_row):
        print(index, column_reader)


# In[15]:


from datetime import datetime
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    dates, highs = [],[]
    for row in reader:
        dates.append(datetime.strptime(row[2],"%Y-%m-%d"))
        highs.append(int(row[5]))
    print(highs,dates)


# In[17]:


import matplotlib.pyplot as plt
fig = plt.figure(dpi=128, figsize=(10,6))
plt.plot(dates, highs, c='red')
plt.title("Daily high temperature, July 2018", fontsize=24)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate() #绘制倾斜的标签以免重叠
plt.ylabel('Temperature (F)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()


# In[30]:


filename = r"C:\Users\Mac\Desktop\ehmatthes-pcc_2e-1.1-0-gbc387ba\ehmatthes-pcc_2e-bc387ba\chapter_16\the_csv_file_format\data\sitka_weather_2018_simple.csv"
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    dates, highs, lows = [], [], []
    for row in reader:
        current_date = datetime.strptime(row[2],"%Y-%m-%d")
        dates.append(current_date)
        high = int(row[5])
        highs.append(high)
        low = int(row[6])
        lows.append(low)
# 根据数据绘制图形
fig = plt.figure(dpi=128, figsize=(10, 6))
plt.plot(dates, highs, c='red')
plt.plot(dates, lows, c='blue')
plt.fill_between(dates,highs, lows, facecolor='blue', alpha=0.1)
# 设置图形的格式
plt.title("Daily high and low temperatures - 2014", fontsize=24)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate() #绘制倾斜的标签以免重叠
plt.ylabel('Temperature (F)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()


# In[ ]:




