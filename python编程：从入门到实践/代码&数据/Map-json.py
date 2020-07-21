#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
filename = r"C:\Users\Mac\Desktop\pcc-master\pcc-master\chapter_16\pygal2_update\population_data.json"
with open(filename) as f:
    pop_data = json.load(f)
for pop_dict in pop_data:
    if pop_dict['Year']=='2010':
        country_name = pop_dict['Country Name']
        population = pop_dict['Value']
        print(country_name,':',population)


# In[1]:


from pygal_maps_world.i18n import COUNTRIES
def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code
    return None
get_country_code('Andorra')


# In[5]:


with open(filename) as f:
    pop_data = json.load(f)
cc_populations = {}
for pop_dict in pop_data:
    if pop_dict['Year']=='2010':
        country_name = pop_dict['Country Name']
        population = int(float(pop_dict['Value']))
        code = get_country_code(country_name)
        if code:
            cc_populations[code] = population
        else:
            pass
            #print(country_name,"has been removed")

# 按人口分组以区分颜色
cc_pop1, cc_pop2, cc_pop3 = {},{},{}
for cc,pop in cc_populations.items():
    if pop < 10**7:
        cc_pop1[cc] = pop
    if pop < 10**9:
        cc_pop1[cc] = pop
    else:
        cc_pop1[cc] = pop


# In[15]:


## 生成交互图像
import pygal
from pygal.style import LightColorizedStyle, RotateStyle
wm_style = RotateStyle('#336699', base_style=LightColorizedStyle)
wm = pygal.maps.world.World(style=wm_style)
wm.title = "World Population in 2010, by Country"
wm.add("0-10m", cc_pop1)
wm.add("10m-1bn", cc_pop2)
wm.add(">1bn", cc_pop3)
#wm.render_to_file('world_population.svg')
wm.render_in_browser()


# In[ ]:




