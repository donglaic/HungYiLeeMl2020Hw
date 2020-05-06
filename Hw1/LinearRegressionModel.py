#!/usr/bin/env python
# coding: utf-8

# In[425]:

import numpy as np

class LinearRegreesionModel:
    '''
    采用gradient descent进行训练的线性模型
    '''
    def __init__(self,lr=0.03,epoches=1000,lbd=0):
        '''
        lr, learning rate
        epoches
        lbd, lambda for regularization
        '''
        self.lr = lr
        self.epoches = epoches
        self.lbd = lbd
        self.theta = None
        self.loss_history = []

    def get_parameters(self):
        return self.theta
        
    def predict(self,X):
        '''
        预测y
        '''
        X_temp = np.concatenate((np.ones((X.shape[0],1),dtype=np.float32),X),axis=1)
        return X_temp@self.theta

    def _error(self,p,y):
        '''
        return error between predict p and target y
        '''
        return p-y
    
    def _loss(self,X,y,lbd=0):
        '''
        计算Mean squared error AND grad
        '''
        if len(X) == 0:
            return 0,0

        X_temp = np.concatenate((np.ones((X.shape[0],1),dtype=np.float32),X),axis=1)
        m = X_temp.shape[0]
        p = X_temp@self.theta
        error = self._error(p,y)
        theta_temp = self.theta.copy()
        theta_temp[0] = 0
        
        loss = ((error.T@error) + lbd*theta_temp.T@theta_temp)/2/m
        loss = loss.item()
        grad = (X_temp.T@error + lbd*theta_temp)/m
        
        return loss,grad
   
    def history(self):
        return self.train_loss_history,self.test_loss_history,self.grad_history

    def noramlEquation(self,X,y):
        X_temp = np.concatenate((np.ones((X.shape[0],1),dtype=np.float32),X),axis=1)
        return np.linalg.inv(X_temp.T@X_temp)@X_temp.T@y
    
    def fit(self,X,y,X_test=[],y_test=[]):
        '''
        fit data by gradient descent
        '''
        self.train_loss_history = []
        self.test_loss_history = []
        self.grad_history = []
        m,n = X.shape
        
        self.theta = np.zeros((n+1,1))
        for epoch in range(self.epoches):
            loss,grad = self._loss(X,y,self.lbd)
            loss_test,_ = self._loss(X_test,y_test)
            self.train_loss_history.append(loss)
            self.test_loss_history.append(loss_test)
            self.grad_history.append(grad)

            self.theta = self.theta - self.lr*grad
            
            if (epoch+1) % 1000 == 0:
                print(f'epoch:{epoch+1} train loss:{loss} test loss:{loss_test}')

            if loss < 1e-6:
                break

        loss,_ = self._loss(X,y,self.lbd)
        self.train_loss_history.append(loss)
        loss_test,_ = self._loss(X_test,y_test)
        self.test_loss_history.append(loss_test)
