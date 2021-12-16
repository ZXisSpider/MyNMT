# -*- coding: utf-8 -*-
class Language:
    sos_token = 0
    eos_token = 1
    pad_token = 2
    unk_token = 3

    def __init__(self, name):
        """
        :param name: 语言名称
        初始化：
        word2index：将单词转换成序号的字典
        word2count：将单词转换为其出现频数
        index2word：将序号转换成为单词
        n_word：index2word字典的长度
        """
        self.name = name
        self.word2index = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>', 3: '<UNK>'}
        self.n_words = len(self.index2word)

    def index_words(self, sentence):
        """
        :param sentence: 输入语句
        :return: 对输入语句的每个单词调用index_word函数
        """
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        """
        :param word: 输入单词
        :return:
            如果单词不在word2index字典中，则在字典中逐个添加单词条目，并在word2count字典中设置好单词频数为1
            如果单词在word2index字典中，则在word2count字典中更新单词频数
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
