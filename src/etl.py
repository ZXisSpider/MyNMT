# -*- coding: utf-8 -*-
import helpers
import torch
from language import Language
from tqdm import tqdm
import math

"""
Data Extraction
"""

max_length = 20

def merge_file(fileA, fileB, fileC):
    """
    合并两个语言文件使其格式符合一行两个句子两种语言，中间用tab隔开
    :param fileA: 文件A
    :param fileB: 文件B
    :return:
    """
    with open(fileA, 'r') as fp1, open(fileB, 'r') as fp2, open(fileC, 'w') as fp3:
        for line1, line2 in tqdm(zip(fp1, fp2)):
            line1 = line1.rstrip()
            line2 = line2.rstrip()
            fp3.write(line1 + '\t' + line2 + '\n')


def train_test_split(file, trainFile, testFile):
    with open(file, 'r') as fp:
        lines = fp.readlines()
    num_lines = len(lines)
    print(f"Number of liens is {num_lines}")
    train_len = math.ceil(0.8 * num_lines)
    train_lines = lines[:train_len]
    test_lines = lines[train_len:]
    with open(trainFile, 'w') as fp:
        for line in tqdm(train_lines):
            fp.write(line)
    with open(testFile, 'w') as fp:
        for line in tqdm(test_lines):
            fp.write(line)

def filter_pair(p):
    """
    :param p: 双语语句对
    :return: 这个语句对是否满足长度都小于max_length的条件，返回Bool值
    """
    is_good_length = len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
    return is_good_length


def filter_pairs(pairs):
    """
    :param pairs: 语句对列表
    :return: 对语句对列表的每个语句对调用filter_pair函数，返回满足条件的语句对列表
    """
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang_name):
    """
    :param lang_name: 语言的名字
    :return: 输入语言的Language类、输出语言的Language类、输入输出语句对
    """
    # Read and filter sentences
    input_lang, output_lang, pairs = read_languages(lang_name)
    pairs = filter_pairs(pairs)

    # Index words
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


def read_languages(lang):
    """
    :param lang: 数据文件名
    :return: 两个语言的Language类，以及从数据文件中提取的双语语句对
    """
    # Read and parse the text file
    doc = open('./data/%s.txt' % lang).read()
    lines = doc.strip().split('\n')

    # Transform the data and initialize language instances
    pairs = [[helpers.normalize_string(s) for s in l.split('\t')] for l in lines]
    input_lang = Language('eng')
    output_lang = Language(lang)

    return input_lang, output_lang, pairs


"""
Data Transformation
"""


# Returns a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    """
    :param lang: 语言对应的Language类
    :param sentence: 句子
    :return: 将句子中的单词转换成对应Language类中的index
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence, device='cpu'):
    """
    :param lang: 对应Language类
    :param sentence: 句子
    :param device: 设备：cpu还是cuda
    :return: 将句子的index序列转换成tensor
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(Language.eos_token)
    while len(indexes) < max_length:
        indexes.append(Language.pad_token)
    tensor = torch.tensor(indexes).view(-1).to(device)
    return tensor


def tensor_from_pair(pair, input_lang, output_lang, device='cpu'):
    """
    :param pair: 语句对
    :param input_lang: 输入语言Language类
    :param output_lang: 输出语言Language类
    :param device: 设备
    :return: 输入句子和输出句子的tensor
    """
    input = tensor_from_sentence(input_lang, pair[0], device)
    target = tensor_from_sentence(output_lang, pair[1], device)
    return input, target


if __name__ == '__main__':
    # merge_file('../data/en-ve/train.en', '../data/en-ve/train.vi', '../data/en-ve/vi.txt')
    train_test_split('../data/en-ve/vi.txt', '../data/en-ve/train.txt', '../data/en-ve/test.txt')