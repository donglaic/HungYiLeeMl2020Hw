#!/usr/bin/python3

from DataLoader import DataLoader
from gensim.models import word2vec

dr = DataLoader()
dr.read()
X_train_label,Y_train_label,X_train_nolabel,X_test = dr.data()

model = word2vec.Word2Vec((X_train_label + X_train_nolabel + X_test),
                          size=250,window=5,min_count=5,workers=12,iter=10,sg=1)

# model = word2vec.Word2Vec((X_train_label + X_test),
#                           size=250,window=5,min_count=5,workers=12,iter=10,sg=1)


model.save('w2v.model')
# model.save('w2v_without_nolabel.model')
