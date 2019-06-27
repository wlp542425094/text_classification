from torch import nn
import gensim
import numpy as np
import torch

class model(nn.Module):
    def __init__(self, batch_size , drop_p=0.5):
        super().__init__()
        # params: "n_" means dimension
        self.batch_size = batch_size
        self.n_layers = 2  # number of LSTM layers
        self.n_hidden = 100  # number of hidden nodes in LSTM

        word2vec =gensim.models.word2vec.Word2Vec.load("word2vec.w2v").wv
        vocab = list(word2vec.wv.vocab.keys())
        vocab_list = []
        for v in vocab:
            vocab_list.append(word2vec[v])
        vocab_numpy = np.array(vocab_list)
        #numpy 中的shape[0]代表行的总数，shape[1]代表列的总数
        self.n_vocab = vocab_numpy.shape[0]
        self.n_embed = vocab_numpy.shape[1]
        #输入的相应的词向量和词的长度
        #这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小，宽是用来表示字典中每个元素的属性向量，向量的维度根据你想要表示的元素的复杂度而定。类实例化之后可以根据字典中元素的下标来查找元素对应的向量。
        self.embedding = nn.Embedding(self.n_vocab, self.n_embed)
        self.embedding.weight.data.copy_(torch.from_numpy(vocab_numpy))

        self.lstm = nn.LSTM(self.n_embed, self.n_hidden, 1, batch_first=True, dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(self.n_hidden, 18)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_words):
        # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_words)  # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)  # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)  # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(lstm_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(self.batch_size, -1)  # (batch_size, seq_length*n_output)

        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]  # (batch_size, 1)

        return sigmoid_last, h



