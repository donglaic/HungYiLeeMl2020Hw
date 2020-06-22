import random
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # input = [batch, seq,    ] = [batch, seq, emb_dim]
        # h_0 = [n_layers*2, batch, hid_dim]
        # output = [seq_len, batch, 2*hid_dim]
        # h_n = [n_layers*2, batch, hid_dim]
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs = [*], output = [*,emb_dim]
        embedding = self.embedding(inputs)
        output, hidden = self.rnn(self.dropout(embedding))
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        #         self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=dropout, batch_first=True)
        # ???? what is this? - it`s a fc
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, encoder_outputs):
        # inputs = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # Decoder 只會是單向，所以 directions=1
        inputs = inputs.unsqueeze(1)
        embedded = self.dropout(self.embedding(inputs))
        # embedded = [batch size, 1, emb dim]
        if self.isatt:
            attn = self.attention(encoder_outputs, hidden)
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
            embedded = torch.cat([embedded, attn], dim=2)
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]

        # 将 RNN 的输出转为每个词出现的机率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, inputs, target, teacher_forcing_ratio):
        # inputs = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少机率使用正确答案来训练
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 准备一个存储空间来存储输出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 将输入放入 Encoder
        encoder_outputs, hidden = self.encoder(inputs)
        # Encoder 最后的隐藏层 hidden state 用来初始化 Decoder
        # encoder outputs 主要使用在 Attention
        # 因为 Encoder 是双向的RNN 所以需要将同一层两个方向的 hidden state 接在一起 ------ ？？？？
        # hidden = [num_layers * directions, batch size, hid dim]
        #          --> [num_layers, directions, batch size, hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        inputs = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(inputs, hidden, encoder_outputs)
            outputs[:, t] = output
            # 决定是否用正确答案来做训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出机率最大的单词
            top1 = output.argmax(1)
            # 如果是 teacher force 则用正解训练 反之用自己预测的单词做预测
            inputs = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    # 这里为什么要给 test 单独一个 inference 而不是共用 train 的 forward 呢 ？？？ 因为 scheduled teaching的缘故
    def inference(self, inputs, target):
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
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        inputs = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(inputs, hidden, encoder_outputs)
            # 将预测结果存起来
            outputs[:, t] = output
            # 取出机率最大的单词
            top1 = output.argmax(1)
            inputs = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions] = [60,50,1024]
        # decoder_hidden = [num_layers, batch size, hid dim] = [3,60,1024]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########

        attention = None
        return attention  # [batch, 1, hid_dim * 2]

if __name__ == '__main__':
    hid_dim = 512
    batch_size = 60
    seq_len = 50
    n_layers = 3

    encoder_outputs = torch.ones((batch_size,seq_len,hid_dim*2))
    decoder_hidden = torch.ones((n_layers,batch_size,hid_dim*2))

    atten = Attention(hid_dim)
    atten(encoder_outputs,decoder_hidden)
    pass