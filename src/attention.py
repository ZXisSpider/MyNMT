# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_size):
        """
        :param method: 计算score的方法
        :param hidden_size: hidden state的大小
        """
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        """
        Attention模块利用decoder上一步输入的hidden state计算与encoder各个output的相似性，并输入softmax函数
        得到attention向量

        :param hidden: decoder上一步输出hidden state
        :param encoder_outputs: encoder的输出，形状为[len, 1, hidden_size]
        :return: attention向量
        """

        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.

        After creating variables to store the attention energies, calculate their
        values for each encoder output and return the normalized values.

        Args:
            hidden: decoder hidden output used for condition  [1, hidden_size]
            encoder_outputs: list of encoder outputs [len, 1, hidden_size]

        Returns:
             Normalized (0..1) energy values, re-sized to 1 x 1 x seq_len
        """
        #  print('hidden: ', hidden.shape)
        #  print('encoder_outputs: ', encoder_outputs.shape)

        batch_size, hidden_size = hidden.size()
        enc_len, batch_size, _ = encoder_outputs.size()
        energies = torch.zeros(batch_size, enc_len).to(encoder_outputs.device)
        for bi in range(batch_size):
            for li in range(enc_len):
                energies[bi, li] = self._score(hidden[bi], encoder_outputs[li, bi, :])
        return F.softmax(energies, dim=0).unsqueeze(1)

    def _score(self, hidden, encoder_output):
        """
        :param hidden: decoder上一步输出hidden state
        :param encoder_outputs: encoder的输出，形状为[len, 1, hidden_size]
        :return: 通过method给出的方法计算向量相似性
        """
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = hidden.view(-1).dot(encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = hidden.view(-1).dot(energy.view(-1))
        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, encoder_output), 1))
            energy = self.other.view(-1).dot(energy.view(-1))
        return energy
