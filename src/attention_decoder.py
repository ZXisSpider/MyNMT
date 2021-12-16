# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention


class AttentionDecoderRNN(nn.Module):
    """
    带attention机制的decoder类
    """
    """Recurrent neural network that makes use of gated recurrent units to translate encoded input using attention."""

    def __init__(self,
                 tgt_vocab_size,
                 embedding_size,
                 hidden_size,
                 attn_model,
                 n_layers=1,
                 dropout=.1):
        """
        :param tgt_vocab_size: 目标语言词汇量
        :param embedding_size: 词嵌入向量维度
        :param hidden_size: 隐状态维度
        :param attn_model: 采用的注意力模型
        :param n_layers: RNN层数
        :param dropout: dropout正则化参数
        """
        super(AttentionDecoderRNN, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_model = attn_model
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, tgt_vocab_size)

        # Choose attention model
        if attn_model is not None:
            self.attention = Attention(attn_model, hidden_size)

    def forward(self, input, decoder_context, hidden_state, encoder_outputs):
        """
        decoder接受上一时刻输出的单词向量与context向量，计算得到下一时刻的隐状态与context向量，并输出单词

        :param input: 输入单词的tensor
        :param decoder_context: 通过attention模块与encoder_outputs计算得到的的的原句context向量
        :param hidden_state: 上一时刻的隐状态
        :param encoder_outputs: encoder得到的输出
        :return:
        output： 预测单词在vocab上的概率分布
        context：这一时刻输出得到的hidden_state与encoder_outputs计算的到的context vector
        hidden_state: 这一时刻输出得到的隐状态
        attention_weights： attention模块计算得到的对encoder各个时刻outputs的重要性权重
        """

        """Run forward propagation one step at a time.

        Get the embedding of the current input word (last output word) [s = 1 x batch_size x seq_len]
        then combine them with the previous context. Use this as input and run through the RNN. Next,
        calculate the attention from the current RNN state and all encoder outputs. The final output
        is the next word prediction using the RNN hidden_state state and context vector.

        Args:
            input: torch Variable representing the word input constituent
            decoder_context: torch Variable representing the previous context
            hidden_state: torch Variable representing the previous hidden_state state output
            encoder_outputs: torch Variable containing the encoder output values

        Return:
            output: torch Variable representing the predicted word constituent
            context: torch Variable representing the context value
            hidden_state: torch Variable representing the hidden_state state of the RNN
            attention_weights: torch Variable retrieved from the attention model
        """

        # Run through RNN
        input = input.view(1, -1)
        embedded = self.embedding(input) # [1, -1, embedding_size]
        embedded = self.dropout(embedded)

        # print(f'Embedded shape is {embedded.shape}')
        # print(f'Decoder_context shape is {decoder_context.shape}')
        rnn_input = torch.cat((embedded, decoder_context), 2) # [1, -1, embedding_size + hidden_size]
        rnn_output, hidden_state = self.gru(rnn_input, hidden_state) # [1, -1, hidden_size]
        # print(f'RNN output is {rnn_output.shape}')
        # print(f'Hidden state is {hidden_state.shape}')

        # Calculate attention
        #  print(rnn_output.shape)
        #  print(encoder_outputs.shape)
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        #  print(attention_weights.shape)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1)) # [-1, 1, hidden_size]
        context = context.transpose(0, 1) # [1, -1, hidden_size]
        # print(f'Context shape is {context.shape}')

        # Predict output
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 2)), dim=2)
        # print(f'Output shape is {output.shape}')
        output = output.squeeze(0)
        return output, context, hidden_state, attention_weights
