#!/usr/bin/env python
# coding: utf-8

# In[11]:


from random import randint
class Die():
    '''定义一个骰子类'''
    def __init__(self, num_sides=6):
        self.num_sides = num_sides
    def roll(self):
        return randint(1, self.num_sides)


# In[1]:


die = Die()
results = []
for roll_num in range(1000):
    result = die.roll()
    results.append(result)
# 分析结果
frequencies = []
for value in range(1, die.num_sides+1):
    frequency = results.count(value)
    frequencies.append(frequency)


# In[17]:


import pygal
hist = pygal.Bar()
hist.title = "Results of rolling one D6 1000 times."
hist.x_labels = ['1', '2', '3', '4', '5', '6']
hist.x_title = "Result"
hist.y_title = "Frequency of Result"
hist.add('D6', frequencies)
hist.render_to_file('die_visual.svg')

