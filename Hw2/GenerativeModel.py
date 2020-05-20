import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return np.clip(1.0/(1.0+np.exp(-1*np.array(z))),1e-8,1-1e-8)

class LogisticRegressionModel:
    
    def __init__(self,lr=0.01,epoches=10,lbd=0,best_accuracy=0,batch_size=8):
        self.lr = lr
        self.epoches = epoches
        self.lbd = lbd
        self.best_accuracy = best_accuracy
        self.batch_size = batch_size

    def _shuffle(self,X,y):
        ids = np.arange(len(X))
        np.random.shuffle(ids)
        return X[ids],y[ids]
    
    def predict(self,X):
        return np.round(self._f(X)).astype(np.int)
    
    def _f(self,X):
        return sigmoid(X@self.theta)
    
    def fit(self,X,y,X_test=[],y_test=[]):
        self.history = {'train loss':[], 'validation loss':[], 'train accuray':[], 'validation accuray':[]}
        
        X = np.concatenate((np.ones((X.shape[0],1),dtype=np.int64),X),axis=1)
        if len(X_test) > 0:
            X_test = np.concatenate((np.ones((X_test.shape[0],1),dtype=np.int64),X_test),axis=1)

        self.theta = np.random.random((X.shape[1],1))
#         self.theta = np.zeros((X.shape[1],1))
        
        step = 1
        ppos = 1
        for epoch in range(self.epoches):
            X,y = self._shuffle(X,y)
            
            for idx in range(int(np.floor(X.shape[0]/self.batch_size))):
                X_train = X[idx*self.batch_size : (idx+1)*self.batch_size]
                y_train = y[idx*self.batch_size : (idx+1)*self.batch_size]

                loss,grad = self._loss(X_train,y_train,lbd=self.lbd)
                self.theta = self.theta - self.lr/(np.sqrt(step))*grad # learning rate decay with time

                step = step + 1

            train_loss,_ = self._loss(X,y)
            test_loss,_ = self._loss(X_test,y_test)
            self.history['train loss'].append(train_loss)
            self.history['validation loss'].append(test_loss)
                
            train_score = self._score(X,y)
            test_score = self._score(X_test,y_test)
            self.history['train accuray'].append(train_score)
            self.history['validation accuray'].append(test_score)
    
            if epoch+1 >= ppos*np.round(self.epoches/10):
                print(f'epoch:{epoch+1} train accuray:{train_score} validation accuray:{test_score}')
                ppos = ppos + 1

            if test_score > self.best_accuracy:
                print(f'better accuracy found:{test_score}, model saved!')
                self.best_accuracy = test_score
                np.save('theta.npy',self.theta)
    
    def _score(self,X,y):
        if len(X) == 0:
            return None
        p = self.predict(X)
        return 1 - np.mean(np.abs(p - y))
    
    def get_history(self):
        return self.history
    
    def _cross_entropy(self,h,y):
        return (-y.T@np.log(h) - (1-y).T@np.log(1-h)).item()
    
    def _loss(self,X,y,lbd=0):
        h = self._f(X)
        theta_temp = self.theta.copy()
        theta_temp[0] = 0
        loss = self._cross_entropy(h,y)/X.shape[0] + (lbd*theta_temp.T@theta_temp/2/X.shape[0]).item()
        grad = X.T@(h-y)/X.shape[0] + lbd/X.shape[0]*theta_temp
        return loss,grad

    def load(self,fn='theta.npy'):
        self.theta = np.load(fn)
                
    def parameters(self):
        return self.theta
