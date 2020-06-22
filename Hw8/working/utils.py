import os
import numpy as np
import re
import json
import torch
import torch.utils.data as data

from model import Encoder,Decoder,Seq2Seq

# 这里定义了一个 transform 来作 padding 很巧妙

class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label

class EN2CNDataset(data.Dataset):
    def __init__(self, root, max_output_len, set_name):
        self.root = root

        # get dictionary
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)

        # load data
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), 'r') as f:
            for l in f:
                self.data.append(l)
        print(f'Loading dataset:{set_name}.txt, len:{len(self.data)}')

        # set transform
        self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])

    def get_dictionary(self, language):
        '''
        get dictionary of expected language
        '''
        with open(os.path.join(self.root, f'word2int_{language}.json'), 'r') as f:
            word2int = json.load(f)
        with open(os.path.join(self.root, f'int2word_{language}.json'), 'r') as f:
            int2word = json.load(f)
        return word2int, int2word

    def split_en_cn_sentences(self, sentence):
        '''
        split en and cn sentences in a sentence
        '''
        sentences = re.split('[\t\n]', sentence)
        sentences = list(filter(None, sentences))  # ignore ''
        assert len(sentences) == 2
        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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
        sentence = re.split(' ', sentence)
        sentence = list(filter(None, sentence))  # ignore ''
        #         print('En sentence: ',sentence)
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        # get cn vector
        # first, split the sentence into subwords and change them into int
        sentence = sentences[1]  # cn sentence
        sentence = re.split(' ', sentence)
        sentence = list(filter(None, sentence))  # ignore ''
        #         print('Cn sentence: ',sentence)
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        # change en and cn to array
        en, cn = np.asarray(en), np.asarray(cn)

        # make the sentences having the same length
        en, cn = self.transform(en), self.transform(cn)

        return en, cn

def save_model(model, store_model_path, step):
    torch.save(model.state_dict(),f'{store_model_path}/model_{step}.ckpt')
    print(f'Saving model {store_model_path}/model_{step}.ckpt')
    return

def load_model(model, load_model_path):
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    print(f'Load model from {load_model_path}')
    return model

def build_model(config,en_vocab_size,cn_vocab_size):
    encoder = Encoder(en_vocab_size,config.emb_dim,config.hid_dim,config.n_layers,config.dropout)
    decoder = Decoder(cn_vocab_size,config.emb_dim,config.hid_dim,config.n_layers,config.dropout,
                      config.attention)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(encoder,decoder,device)
    print(model)
    # 构建 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print('\n',optimizer)
    if config.load_model:
        model = load_model(model,config.load_model_path)
    model = model.to(device)
    return model, optimizer

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


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def computebleu(sentences, targets):
    score = 0
    assert len(sentences) == len(targets)

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            # what is this???
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence, target = cut_token(sentence), cut_token(target)
        score += sentence_bleu([target], sentence, weights=[1, 0, 0, 0])

    return score

def infinit_iter(dataloader):
    it = iter(dataloader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(dataloader)

########
# TODO #
########

# 請在這裡直接 return 0 來取消 Teacher Forcing
# 請在這裡實作 schedule_sampling 的策略

# linear decay
global_schedule_ratio = 1
# def schedule_sampling():
#     global global_schedule_ratio
#     global_schedule_ratio -= 1.0/config.num_steps
#     return global_schedule_ratio

# always uses teacher forcing
# def schedule_sampling():
#     return 1

# inverse sigmoid decay
global_step = 0
# def schedule_sampling():
#     global global_step
#     global_step += 1
#     x = (global_step - config.num_steps/2)/1000
#     return 1 - F.sigmoid(torch.tensor(x,dtype=torch.float)).item()

# exponential decay
def schedule_sampling():
    global global_step
    global_step += 1
    return 1 - np.log(global_step)

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
        self.num_steps = 120
        # 训练多少次后需要存储模型
        self.store_steps = 9
        # 训练多少次后需要检验是否有 overfitting
        self.summary_steps = 3
        # 是否需要载入模型
        self.load_model = False
        # 存储模型的位置
        self.store_model_path = '.'
        # 载入模型的位置
        self.load_model_path = f'{self.store_model_path}/model_{self.num_steps}'
        # 资料存放的位置
        self.data_path = '../input/hw8-data/cmn-eng'
        # 是否使用 Attention Mechanism
        self.attention = True