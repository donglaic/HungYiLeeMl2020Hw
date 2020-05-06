#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd


# In[46]:


class DataReader:
    def __init__(self,fileName='train.csv',hours=9,
                 features=['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 
                           'NOx', 'O3', 'PM10','PM2.5', 'RAINFALL', 
                           'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC','WIND_SPEED', 'WS_HR']):
        self.fileName = fileName
        self.hours = hours
        self.features = features
    
    def _readDfData(self):
        '''
        读取DataFrame数据，并处理NA值
        '''
        data = pd.read_csv(self.fileName,engine='python')
        data.columns = ['Date', 'Place', 'Test', '0', '1', '2', '3', '4', '5', '6', '7', '8',
               '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', '23']
        data.drop(['Date','Place'],axis='columns',inplace=True)
        data.set_index('Test',inplace=True)
        data[data=='NR'] = 0 # 处理NA值
        return data

    def _get_data_dict(self,data):
        '''
        将df数据存成dict，key为月序号，data为df类型的数据
        '''
        data_dict = {}
        for m in range(12):
            row = m*18*20
            temp = data.iloc[row:row+18]
            for d in range(1,20):
                row += 18
                temp = pd.concat([temp,data.iloc[row:row+18]],axis=1)
            temp.set_axis(range(24*20),axis=1,inplace=True)
            data_dict[m] = temp
        return data_dict

    def _getMonthTrainData(self,month,hours,features):
        label = month.loc['PM2.5',hours:]
        X = np.empty((480-hours,len(features)*hours))
        for i in range(480-hours):
            X[i,:] = month.loc[features,i:i+hours-1].values.flatten()
        return X,label.values.astype(np.float)

    def data(self):
        '''
        转换后的train data
        return X, ndarray, train data
               y, ndarray, label data 
        '''
        dict = self._get_data_dict(self._readDfData())
        X,y = self._getMonthTrainData(dict[0],self.hours,self.features)
        for m in range(1,12):
            temp_X,temp_y = self._getMonthTrainData(dict[m],self.hours,self.features)
            X = np.concatenate((X,temp_X),axis=0)
            y = np.concatenate((y,temp_y),axis=0)
        return X,y.reshape(y.shape[0],-1)