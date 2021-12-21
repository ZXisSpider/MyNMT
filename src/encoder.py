# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    Encoder类：将输入语句根据时序信息编码
    """
    """Recurrent neural network that encodes a given input sequence."""

    def __init__(self, batch_size, src_vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1):
        """
        :param batch_size: batch大小
        :param src_vocab_size: 源语言单词总数
        :param embedding_size: word embedding大小
        :param hidden_size: 隐状态大小
        :param n_layers: RNN层数
        :param dropout: dropout参数
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.src_vocab_size = src_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, inputs, hidden_state):
        """
        :param inputs: 输入语句tensor
        :param hidden_state: 初始hidden state
        :return: 返回GRU的output与hidden state
        """
        inputs = inputs.view(-1, self.batch_size)
        embedded = self.embedding(inputs) # [len, batch_size, embedding_size]
        embedded = self.dropout(embedded)
        output, hidden_state = self.rnn(embedded, hidden_state)
        return output, hidden_state

    def init_hidden(self, device):
        """
        :param device: 设备 CPU/CUDA
        :return: 初始隐状态
        """
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device)
        return hidden_state
