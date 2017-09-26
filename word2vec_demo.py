# !/usr/bin/env python
# -*- coding: utf-8 -*-

from nnclf.word_vector import WordVector
import argparse
import os

parser = argparse.ArgumentParser(description="word2vec")
parser.add_argument('-word', type=str, default='贷款', help='input word')
parser.add_argument('-train', action='store_true', default=False, help='train')

args = parser.parse_args()
w2v = WordVector('./runtime/word2vec.bin')

if args.train:
    import jieba
    fw = open('./corpus/data.txt', 'w')
    corpus_dir = './corpus/data'
    for filename in os.listdir(corpus_dir):
        with open(os.path.join(corpus_dir, filename), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                fw.writelines(' '.join(jieba.lcut(line)) + '\n')
    fw.close()
    w2v.train('./corpus/data.txt')

model = w2v.load()
string = args.word
if string not in model:
    print('当前词不在词库中')
    exit()
print('当前词')
print(string)
print('词向量')
print(model[string])
print('近义词(余弦定理)：')
indexes, metrics = model.cosine(string)
print(model.generate_response(indexes, metrics).tolist())