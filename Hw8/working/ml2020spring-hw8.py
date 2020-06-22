#!/usr/bin/env python
# coding: utf-8

# # Homework 8 - Sequence-to-sequence
# 
# # Sequence-to-Sequence 介紹
# - 大多數常見的 **sequence-to-sequence (seq2seq) model** 為 **encoder-decoder model**，主要由兩個部分組成，分別是 **Encoder** 和 **Decoder**，而這兩個部分則大多使用 **recurrent neural network (RNN)** 來實作，主要是用來解決輸入和輸出的長度不一樣的情況
# - **Encoder** 是將**一連串**的輸入，如文字、影片、聲音訊號等，編碼為**單個向量**，這單個向量可以想像為是整個輸入的抽象表示，包含了整個輸入的資訊
# - **Decoder** 是將 Encoder 輸出的單個向量逐步解碼，**一次輸出一個結果**，直到將最後目標輸出被產生出來為止，每次輸出會影響下一次的輸出，一般會在開頭加入 "< BOS >" 來表示開始解碼，會在結尾輸出 "< EOS >" 來表示輸出結束
# 
# 
# ![seq2seq](https://i.imgur.com/0zeDyuI.png)
# 
# # 作業介紹
# - 英文翻譯中文
#   - 輸入： 一句英文 （e.g.		tom is a student .） 
#   - 輸出： 中文翻譯 （e.g. 		湯姆 是 個 學生 。）
# 
# - TODO
#   - Teachering Forcing 的功用: 嘗試不用 Teachering Forcing 做訓練
#   - 實作 Attention Mechanism
#   - 實作 Beam Search
#   - 實作 Schedule Sampling
# 
# # 资料下载

# In[7]:



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import sys
import os
import random
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算


# # 资料结构
# ## 定义资料的转换
# - 将不同长度的答案拓展到相同长度，以便训练模型

# In[8]:


# 这里定义了一个 transform 来作 padding 很巧妙

class LabelTransform(object):
    def __init__(self,size,pad):
        self.size = size
        self.pad = pad
    
    def __call__(self,label):
        label = np.pad(label,(0,(self.size - label.shape[0])),mode='constant',constant_values=self.pad)
        return label


# ## 定義 Dataset
# - Data (出自manythings 的 cmn-eng):
#   - 訓練資料：18000句
#   - 檢驗資料：  500句
#   - 測試資料： 2636句
# 
# - 資料預處理:
#   - 英文：
#     - 用 subword-nmt 套件將word轉為subword
#     - 建立字典：取出標籤中出現頻率高於定值的subword
#   - 中文：
#     - 用 jieba 將中文句子斷詞
#     - 建立字典：取出標籤中出現頻率高於定值的詞
#   - 特殊字元： < PAD >, < BOS >, < EOS >, < UNK > 
#     - < PAD >  ：無意義，將句子拓展到相同長度
#     - < BOS >  ：Begin of sentence, 開始字元
#     - < EOS >  ：End of sentence, 結尾字元
#     - < UNK > ：單字沒有出現在字典裡的字
#   - 將字典裡每個 subword (詞) 用一個整數表示，分為英文和中文的字典，方便之後轉為 one-hot vector   
# 
# - 處理後的檔案:
#   - 字典：
#     - int2word_*.json: 將整數轉為文字
#     ![int2word_en.json](https://i.imgur.com/31E4MdZ.png)
#     - word2int_*.json: 將文字轉為整數
#     ![word2int_en.json](https://i.imgur.com/9vI4AS1.png)
#     - $*$ 分為英文（en）和中文（cn）
#   
#   - 訓練資料:
#     - 不同語言的句子用 TAB ('\t') 分開
#     - 字跟字之間用空白分開
#     ![data](https://i.imgur.com/nSH1fH4.png)
# 
# 
# - 在將答案傳出去前，在答案開頭加入 "< BOS >" 符號，並於答案結尾加入 "< EOS >" 符號

# In[9]:


import re
import json

class EN2CNDataset(data.Dataset):
    def __init__(self,root,max_output_len,set_name):
        self.root = root
        
        # get dictionary
        self.word2int_cn,self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en,self.int2word_en = self.get_dictionary('en')
        
        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)
        
        # load data
        self.data = []
        with open(os.path.join(self.root,f'{set_name}.txt'),'r') as f:
            for l in f:
                self.data.append(l)
        print(f'Loading dataset:{set_name}.txt, len:{len(self.data)}')
        
        # set transform
        self.transform = LabelTransform(max_output_len,self.word2int_en['<PAD>'])
        
    def get_dictionary(self,language):
        '''
        get dictionary of expected language
        '''
        with open(os.path.join(self.root,f'word2int_{language}.json'),'r') as f:
            word2int = json.load(f)
        with open(os.path.join(self.root,f'int2word_{language}.json'),'r') as f:
            int2word = json.load(f)
        return word2int, int2word
    
    def split_en_cn_sentences(self,sentence):
        '''
        split en and cn sentences in a sentence
        '''
        sentences = re.split('[\t\n]',sentence)
        sentences = list(filter(None,sentences)) # ignore ''
        assert len(sentences) == 2
        return sentences

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        '''
        get item of desired index
        return en,cn - en and cn sentence, LongTensor, same length
        '''
        # split en cn sentences
        sentences = self.split_en_cn_sentences(self.data[index])
        
        # special words
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']
        
        # get en and cn vectors, all begin with <BOS>, endwith <EOS>
        en, cn = [BOS], [BOS]
        
        # get en vector
        # first, split the sentence into subwords and change them into int
        sentence = sentences[0]
        sentence = re.split(' ',sentence)
        sentence = list(filter(None,sentence)) # ignore ''
#         print('En sentence: ',sentence)
        for word in sentence:
            en.append(self.word2int_en.get(word,UNK))
        en.append(EOS)
        
        # get cn vector
        # first, split the sentence into subwords and change them into int
        sentence = sentences[1] # cn sentence
        sentence = re.split(' ',sentence)
        sentence = list(filter(None,sentence)) # ignore ''
#         print('Cn sentence: ',sentence)
        for word in sentence:
            cn.append(self.word2int_cn.get(word,UNK))
        cn.append(EOS)
        
        # change en and cn to array
        en,cn = np.asarray(en),np.asarray(cn)
        
        # make the sentences having the same length
        en,cn = self.transform(en), self.transform(cn)
        
        return en,cn


# # 模型架构
# 
# ## Encoder
# - seq2seq模型的編碼器為RNN。 對於每個輸入，，**Encoder** 會輸出**一個向量**和**一個隱藏狀態(hidden state)**，並將隱藏狀態用於下一個輸入，換句話說，**Encoder** 會逐步讀取輸入序列，並輸出單個矢量（最終隱藏狀態）
# - 參數:
#   - en_vocab_size 是英文字典的大小，也就是英文的 subword 的個數
#   - emb_dim 是 embedding 的維度，主要將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
#   - hid_dim 是 RNN 輸出和隱藏狀態的維度
#   - n_layers 是 RNN 要疊多少層
#   - dropout 是決定有多少的機率會將某個節點變為 0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不使用
# - Encoder 的輸入和輸出:
#   - 輸入: 
#     - 英文的整數序列 e.g. 1, 28, 29, 205, 2
#   - 輸出: 
#     - outputs: 最上層 RNN 全部的輸出，可以用 Attention 再進行處理
#     - hidden: 每層最後的隱藏狀態，將傳遞到 Decoder 進行解碼
# 

# In[10]:


class Encoder(nn.Module):
    def __init__(self,en_vocab_size,emb_dim,hid_dim,n_layers,dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size,emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
            # input = [batch, seq,    ] = [batch, seq, emb_dim]
        # h_0 = [n_layers*2, batch, hid_dim]
        # output = [seq_len, batch, 2*hid_dim]
        # h_n = [n_layers*2, batch, hid_dim]
        self.rnn = nn.GRU(emb_dim,hid_dim,n_layers,dropout=dropout,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,inputs):
        # inputs = [*], output = [*,emb_dim]
        embedding = self.embedding(inputs)
        # embedding = [batch, seq, emb_dim]
        output, hidden = self.rnn(self.dropout(embedding))
        # output = [batch, seq, hid_dim * direction]
        # hidden = [n_layers*directions, batch, hid_dim]
        return output, hidden


# # Decoder
# - **Decoder** 是另一個 RNN，在最簡單的 seq2seq decoder 中，僅使用 **Encoder** 每一層最後的隱藏狀態來進行解碼，而這最後的隱藏狀態有時被稱為 “content vector”，因為可以想像它對整個前文序列進行編碼， 此 “content vector” 用作 **Decoder** 的**初始**隱藏狀態， 而 **Encoder** 的輸出通常用於 Attention Mechanism
# - 參數
#   - cn_vocab_size 是英文字典的大小，也就是中文的 subword 的個數
#   - emb_dim 是 embedding 的維度，是用來將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
#   - hid_dim 是 RNN 輸出和隱藏狀態的維度
#   - output_dim 是最終輸出的維度，一般來說是將 hid_dim 轉到 one-hot vector 的單詞向量
#   - n_layers 是 RNN 要疊多少層
#   - dropout 是決定有多少的機率會將某個節點變為0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不用
#   - isatt 是來決定是否使用 Attention Mechanism
# 
# - Decoder 的輸入和輸出:
#   - 輸入:
#     - 前一次解碼出來的單詞的整數表示
#   - 輸出:
#     - hidden: 根據輸入和前一次的隱藏狀態，現在的隱藏狀態更新的結果
#     - output: 每個字有多少機率是這次解碼的結果

# In[11]:


class Decoder(nn.Module):
    def __init__(self,cn_vocab_size,emb_dim,hid_dim,n_layers,dropout,isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size,emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        # self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim,self.hid_dim,self.n_layers,dropout=dropout,batch_first=True)
        # ???? what is this? - it`s a fc
        self.embedding2vocab1 = nn.Linear(self.hid_dim,self.hid_dim*2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim*2,self.hid_dim*4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim*4,self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,inputs,hidden, encoder_outputs):
        # inputs = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # Decoder 只會是單向，所以 directions=1
        inputs = inputs.unsqueeze(1)  # why ??? , 因为 GRU 期望输入为三维 即 【batch, seq, input_size], 而现在的 inputs 是
        # 将targets切片生成的，少了一个 seq 纬度。为什么没有一次性将所有 seq 输入 GRU 呢？ 因为要做 scheduled teaching啊
        embedded = self.dropout(self.embedding(inputs))
        # embedded = [batch size, 1, emb dim]
        if self.isatt:
            attn = self.attention(encoder_outputs,hidden)
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
        output, hidden = self.rnn(embedded,hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]
        
        # 将 RNN 的输出转为每个词出现的机率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden


# ## Attention
# - 當輸入過長，或是單獨靠 “content vector” 無法取得整個輸入的意思時，用 Attention Mechanism 來提供 **Decoder** 更多的資訊
# - 主要是根據現在 **Decoder hidden state** ，去計算在 **Encoder outputs** 中，那些與其有較高的關係，根據關系的數值來決定該傳給 **Decoder** 那些額外資訊 
# - 常見 Attention 的實作是用 Neural Network / Dot Product 來算 **Decoder hidden state** 和 **Encoder outputs** 之間的關係，再對所有算出來的數值做 **softmax** ，最後根據過完 **softmax** 的值對 **Encoder outputs** 做 **weight sum**
# 
# - TODO:
# 實作 Attention Mechanism

# In[12]:


class Attention(nn.Module):
    def __init__(self,hid_dim):
        super(Attention,self).__init__()
        self.hid_dim = hid_dim
        
    def forward(self,encoder_outputs,decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        attention=None

        return attention


# ## Seq2Seq
# - 由 **Encoder** 和 **Decoder** 組成
# - 接收輸入並傳給 **Encoder** 
# - 將 **Encoder** 的輸出傳給 **Decoder**
# - 不斷地將 **Decoder** 的輸出傳回 **Decoder** ，進行解碼  
# - 當解碼完成後，將 **Decoder** 的輸出傳回 

# In[13]:


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers,             "Encoder and decoder must have equal number of layers!"
        
    def forward(self,inputs,target,teacher_forcing_ratio):
        # inputs = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少机率使用正确答案来训练
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size
        
        # 准备一个存储空间来存储输出
        outputs = torch.zeros(batch_size,target_len,vocab_size).to(self.device)
        # 将输入放入 Encoder
        encoder_outputs, hidden = self.encoder(inputs)
        # Encoder 最后的隐藏层 hidden state 用来初始化 Decoder
        # encoder outputs 主要使用在 Attention
        # 因为 Encoder 是双向的RNN 所以需要将同一层两个方向的 hidden state 接在一起 ------ ？？？？
        # hidden = [num_layers * directions, batch size, hid dim] 
        #          --> [num_layers, directions, batch size, hid dim]
        #           ---> [num_layers, batch, hid_di*directions]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:,-2,:,:], hidden[:,-1,:,:]),dim=2)
        # 取的 <BOS> token
        inputs = target[:,0]
        preds = []
        for t in range(1,target_len):
            output, hidden = self.decoder(inputs,hidden,encoder_outputs)
            outputs[:,t] = output
            # 决定是否用正确答案来做训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出机率最大的单词
            top1 = output.argmax(1)
            # 如果是 teacher force 则用正解训练 反之用自己预测的单词做预测
            inputs = target[:,t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds
    
    # 这里为什么要给 test 单独一个 inference 而不是共用 train 的 forward 呢 ？？？
    def inference(self,inputs,target):
        #########
        # TODO  #
        #########
        # 在这里实施 Beam Search
        # 此函数的 batch size = 1
        # input = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = inputs.shape[0]
        input_len = inputs.shape[1]
        vocab_size = self.decoder.cn_vocab_size
        
        # 准备一个存储空间来存储输出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 将输入放入 Encoder
        encoder_outputs, hidden = self.encoder(inputs)
        # Encoder 最后的隐藏层 hidden state 用来初始化 Decoder
        # encoder ouputs 主要用在 Attention
        # 因为 Encoder 是双向的 RNN 所以需要将同一层两个方向的 hidden state 接在一起
        # hidden = [num_layers * directions, batch_size, hid_dim] 
        #           --> [num_layers, direction, batch_size, hid_dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:,-2,:,:],hidden[:,-1,:,:]),dim=2)
        # 取的 <BOS> token
        inputs = target[:,0]
        preds = []
        for t in range(1, input_len):
            output,hidden = self.decoder(inputs,hidden,encoder_outputs)
            # 将预测结果存起来
            outputs[:,t] = output
            # 取出机率最大的单词
            top1 = output.argmax(1)
            inputs = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds,1)
        return outputs, preds


# # utils
# - 基本操作:
#   - 儲存模型
#   - 載入模型
#   - 建構模型
#   - 將一連串的數字還原回句子
#   - 計算 BLEU score
#   - 迭代 dataloader
#   

# ## 存储模型

# In[14]:


def save_model(model, store_model_path, step):
    torch.save(model.state_dict(),f'{store_model_path}/model_{step}.ckpt')
    print(f'Saving model {store_model_path}/model_{step}.ckpt')
    return


# ## 载入模型

# In[15]:


def load_model(model, load_model_path):
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    print(f'Load model from {load_model_path}')
    return model


# ## 构建模型

# In[16]:


def build_model(config,en_vocab_size,cn_vocab_size):
    encoder = Encoder(en_vocab_size,config.emb_dim,config.hid_dim,config.n_layers,config.dropout)
    decoder = Decoder(cn_vocab_size,config.emb_dim,config.hid_dim,config.n_layers,config.dropout,
                      config.attention)
    model = Seq2Seq(encoder,decoder,device)
    print(model)
    # 构建 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print('\n',optimizer)
    if config.load_model:
        model = load_model(model,config.load_model_path)
    model = model.to(device)
    return model, optimizer


# ## 数字转句子

# In[17]:


def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


# ## 计算 BLUE score

# In[18]:


import nltk
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

def computebleu(sentences,targets):
    score = 0
    assert len(sentences) == len(targets)
    
    def cut_token(sentence):
        tmp = []
        for token in sentence:
            # what is this???
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0],encoding='utf-8'))==1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp
    
    for sentence,target in zip(sentences,targets):
        sentence,target= cut_token(sentence),cut_token(target)
        score += sentence_bleu([target],sentence,weights=[1,0,0,0])
    
    return score
        


# ## 迭代 dataloader

# In[19]:


def infinit_iter(dataloader):
    it = iter(dataloader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(dataloader)        


# ## schedule sampling

# In[20]:


########
# TODO #
########

# 請在這裡直接 return 0 來取消 Teacher Forcing
# 請在這裡實作 schedule_sampling 的策略

def schedule_sampling():
    return 1


# # 训练步骤
# ## 训练
# - 训练阶段

# In[21]:


def train(model,optimizer,train_iter,loss_function,total_steps,summary_steps):
    '''
    return model, optimizer, losses
    '''
    # set train in train model
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources,targets = next(train_iter)
        sources,targets = sources.to(device),targets.to(device)
        outputs,preds = model(sources,targets,schedule_sampling())
        # targets 的第一个 toekn 是 <BOS> 所以忽略
        outputs = outputs[:,1:].reshape(-1,outputs.size(2))
        targets = targets[:,1:].reshape(-1)
        loss = loss_function(outputs,targets)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1) # why ??? 解决梯度爆炸的问题
        optimizer.step()
        
        loss_sum += loss.item()
        
        if (step+1) % 5 == 0:
            loss_sum = loss_sum/5
            print('\r','train [{}] loss {:.3f}, Perplexity {:.3f}    '.format(
                total_steps + step + 1,   loss_sum,   np.exp(loss_sum)
                ),end=' ')
            losses.append(loss_sum)
            loss_sum = 0.0
        
    return model,optimizer,losses


# ## 检验/测试
# - 防止训练发生 overfitting

# In[22]:


def test(model,dataloader,loss_function):
    model.eval()
    loss_sum, bleu_score = 0.0,0.0
    n = 0
    result = []
    for sources, targets in dataloader:
        sources,targets = sources.to(device),targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets)
        # targets 的第一个 token 是 <BOS> 所以忽略
        outputs = outputs[:,1:].reshape(-1, outputs.size(2))
        targets = targets[:,1:].reshape(-1)
        
        loss = loss_function(outputs,targets)
        loss_sum += loss.item()
        
        # 将预测结果转为文字
        targets = targets.view(sources.size(0),-1)
        preds = tokens2sentence(preds,dataloader.dataset.int2word_cn)
        sources = tokens2sentence(sources,dataloader.dataset.int2word_en)
        targets = tokens2sentence(targets,dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source,pred,target))
        # 计算 Bleu Score
        bleu_score += computebleu(preds, targets)
        
        n += batch_size
        
    return loss_sum / len(dataloader), bleu_score/n, result    


# ## 训练流程
# - 先训练 再检验

# In[32]:


def train_process(config):
    start = time.time()
    # 准备训练资料
    train_dataset = EN2CNDataset(config.data_path,config.max_output_len,'training')
    train_loader = data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    train_iter = infinit_iter(train_loader)
    # 准备检验资料
    val_dataset = EN2CNDataset(config.data_path,config.max_output_len,'validation')
    val_loader = data.DataLoader(val_dataset,batch_size=1)
    # 构建模型
    model, optimizer = build_model(config,train_dataset.en_vocab_size,train_dataset.cn_vocab_size)  
    loss_function = nn.CrossEntropyLoss(ignore_index=0) # ？？？ 是 ignore bias 的 grad 吗

    # 训练过程
    train_loss, val_losses, bleu_scores = [],[],[]
    total_steps = 0
    while total_steps < config.num_steps:
        # 训练模型
        model, optimizer, losses = train(model,optimizer,train_iter,loss_function,total_steps,
             config.summary_steps)
        train_loss += losses
        # 检验模型
        val_loss, bleu_score,result = test(model,val_loader,loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        
        total_steps += config.summary_steps
        print('\r','val [{}] loss {:.3f}, Perplexity: {:.3f}, bleu score: {:.3f}, used {} seconds      '.format(
                total_steps, val_loss, np.exp(val_loss), bleu_score, int(time.time()-start)
            ))
        
        # 储存模型和结果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model,config.store_model_path,total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt','w') as f:
                for l in result:
                    print(l,file=f)
    
    return train_loss, val_losses, bleu_scores


# ## 测试流程

# In[33]:


def test_process(config):
    # 准备测试资料
    test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
    test_loader = data.DataLoader(test_dataset,batch_size=1)
    # 建构模型
    model, optimizer = build_model(config,test_dataset.en_vocab_size,test_dataset.cn_vocab_size)
    print('Finish build model')
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 测试模型
    test_loss, bleu_score, result = test(model, test_loader, loss_function)
    # 储存结果
    with open(f'./test_output.txt','w') as f:
        for line in result:
            print(line, file=f)
            
    return test_loss, bleu_score


# ## Config
# - 实验的参数设定表

# In[34]:


class Configuration(object):
    def __init__(self):
        self.batch_size = 60
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.learning_rate = 0.00005
        # 最后输出句子的最大长度 
        self.max_output_len = 50
        # 总训练次数
        self.num_steps = 12000
        # 训练多少次后需要存储模型
        self.store_steps = 900
        # 训练多少次后需要检验是否有 overfitting
        self.summary_steps = 300
        # 是否需要载入模型
        self.load_model = False
        # 存储模型的位置
        self.store_model_path = '.'
        # 载入模型的位置
        self.load_model_path = f'{self.store_model_path}/model_{self.num_steps}'
        # 资料存放的位置
        self.data_path = '../input/cmn-eng'
        # 是否使用 Attention Mechanism
        self.attention = False


# # Main Function
# - 载入参数
# - 进行训练 or 推论
# 
# ## Train

# In[42]:


if __name__ == '__main__':
    config = Configuration()
    print('config:\n',vars(config),'\n')
    train_losses, val_losses, bleu_scores = train_process(config)


# ## Test

# In[ ]:


# 在执行 test 之前 需要在 config 中设定要载入的模型位置
if __name__ == '__main__':
    config = Configuration()
    config.load_model = True
    print('config:\n',vars(config),'\n')
    test_loss, bleu_score = test_process(config)
    print(f'test loss: {test_loss}, bleu_score: {bleu_score}')


# # 图形化训练过程
# ## 以图表呈现训练的loss的变化趋势

# In[ ]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_losses)
plt.xlabel('次数')
plt.ylabel('loss')
plt.title('train loss')
plt.show()


# ## 以图表呈现检验的loss变化趋势

# In[ ]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(val_losses)
plt.xlabel('次数')
plt.ylabel('loss')
plt.title('validation loss')
plt.show()


# ## BLEU scroe

# In[ ]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(bleu_scores)
plt.xlabel('次数')
plt.ylabel('BLEU score')
plt.title('BLEU score')
plt.show()

