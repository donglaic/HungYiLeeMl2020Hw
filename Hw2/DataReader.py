#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataReader:
    '''
    数据读取
    '''
    def __init__(self,X_train_fn='X_train',Y_train_fn='Y_train',X_test_fn='X_test'
                 ,specified_columns=None,plynomial=1,val_size=0):
        self.X_train_fn = X_train_fn
        self.Y_train_fn = Y_train_fn
        self.X_test_fn = X_test_fn
        self.specified_columns = specified_columns
        self.plynomial = plynomial
        self.val_size = val_size

    def read(self):
        '''
        read data
        '''
        def read_df(fn):
            df = pd.read_csv(fn)
            df.set_index('id',inplace=True)
            return df
        X_train_df = read_df(self.X_train_fn)
        X = X_train_df.to_numpy(dtype=np.float32)
        Y = read_df(self.Y_train_fn).to_numpy(dtype=np.float32)
        
        if self.val_size > 0:
            self.X_train,self.X_val,self.Y_train,self.Y_val = train_test_split(X,Y,random_state=42,test_size=self.val_size)
        elif self.val_size == 0:
            self.X_train,self.Y_train = (X,Y)
            self.X_val,self.Y_val = (None,None)
        
        self.columns = list(X_train_df.columns)
        self.X_test = read_df(self.X_test_fn).to_numpy(dtype=np.float32)

        if self.plynomial > 1:
            for i in range(2,self.plynomial+1):                
                temp = self.X_train**self.plynomial
                self.X_train = np.concatenate((self.X_train,temp),axis=1)
                temp = self.X_test**self.plynomial
                self.X_test = np.concatenate((self.X_test,temp),axis=1)

        self.X_train = self.normalize(self.X_train)
        if self.val_size > 0:
            self.X_val = self.normalize(self.X_val,train=False)
        self.X_test = self.normalize(self.X_test,train=False)
        
    def data(self):
        '''
        return data
        X_train, Y_train, X_test
        '''
        return self.X_train,self.X_val,self.X_test,self.Y_train,self.Y_val

    def featuresName(self,index):
        return self.columns[index]

    def normalize(self,X,train=True):
        if self.specified_columns == None:
            specified_columns = np.arange(X.shape[1])
        else:
            specified_columns = self.specified_columns
        if train:
            self.X_mean = np.mean(X[:,specified_columns],axis=0).reshape(1,-1)
            self.X_std = np.std(X[:,specified_columns],axis=0).reshape(1,-1)
            
        X[:,specified_columns] = (X[:,specified_columns] - self.X_mean) / (self.X_std + 1e-8)
    
        return X
    

if __name__ == '__main__':
    dr = DataReader()
    dr.read()
    X_train, X_val, X_test, y_train, y_val = dr.data()
    print('ok')