#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
X = np.random.randn(50,5)
y1 = np.random.randn(50,1)
y2 = np.random.randint(0,2,50)


# In[4]:


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
reg.fit(X, y1)
clf.fit(X, y2)


# In[5]:


import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(reg, out_file=None, feature_names=list('abcde'), filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(dot_data)
graph


# In[6]:


dot_data = export_graphviz(clf, out_file=None, feature_names=list('abcde'), filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(dot_data)
graph


# In[ ]:




