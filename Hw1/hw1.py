#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# In[103]:


parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')

a = parser.parse_args()


# read data


data = pd.read_csv(a.input_file,engine='python',header=None)

data.columns = ['id', 'Test', '0', '1', '2', '3', '4', '5', '6', '7', '8']

data.set_index('Test',inplace=True)

data[data=='NR'] = 0

# data.drop(['id','0','1','2','3'],axis='columns',inplace=True)
data.drop(['id'],axis='columns',inplace=True)

data.head()


# In[67]:


m = np.uint32(data.shape[0]/18)


# In[68]:


X = data.values.astype(np.float).reshape(m,-1)

# In[69]:


theta = np.load('theta.npy')

mu = np.load('mu.npy')

sigma = np.load('sigma.npy')

X_norm = (X-mu)/sigma

# In[97]:

X_norm = np.concatenate((np.ones((m,1),dtype=np.float),X_norm),axis=1)

y = X_norm@theta


# In[98]:


d = {'id':[f'id_{i}' for i in range(240)],'value':y.reshape(m,)}


# In[100]:


out = pd.DataFrame(d)


# In[102]:


out.to_csv(a.output_file,index=False)


# In[ ]:




