#!/usr/bin/env python
# coding: utf-8

# In[1]:


from DataReader import DataReader

dr = DataReader(hours=9)
X,y = dr.data()


# In[2]:


import numpy as np


# In[3]:


X = np.concatenate((X,(X**2),(X**3)),axis=1)


# In[4]:


def normal(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X-mu)/std
    return X_norm,mu,std


# In[5]:


from LinearRegressionModel import LinearRegreesionModel


# In[6]:


X_norm,mu,std = normal(X)


# In[7]:


model = LinearRegreesionModel(lr=0.01,lbd=100,epoches=50000)
model.fit(X_norm,y)
train_loss_history,test_loss_history,grad_history = model.history()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(train_loss_history[200:])

print('loss',train_loss_history[-1])


# In[ ]:


np.save('theta.npy',model.get_parameters())

np.save('mu.npy',mu)

np.save('sigma.npy',std)


# In[ ]:


theta = np.load('theta.npy')
mu = np.load('mu.npy')
std = np.load('sigma.npy')


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('test.csv',header=None)
df[df=='NR'] = 0
df.drop([0,1],axis=1,inplace=True)
# df.head(18)


# In[ ]:


test_data = np.array(df)


# In[ ]:


test_data = test_data.astype(np.float32).reshape(240,-1)


# In[ ]:


test_data = np.concatenate((test_data,(test_data**2),(test_data**3)),axis=1)


# In[ ]:


test_data = (test_data-mu)/std


# In[ ]:


test_data = np.concatenate((np.ones((240,1),dtype=np.float32),test_data),axis=1)


# In[ ]:


p = test_data@theta


# In[ ]:


out_df = pd.DataFrame({'id':[f'id_{i}' for i in range(240)],'value':p.flatten()})


# In[ ]:


out_df.to_csv('result.csv',index=False)

