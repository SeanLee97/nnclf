# !/usr/bin/env python
# -*- coding: utf-8 -*-

import word2vec

import os, codecs
import jieba
import numpy as np

class WordVector(object):
        def __init__(self, model_path):
                self.model_path = model_path

        def train(self, corpuspath):
                global word2vec
                word2vec.word2vec(corpuspath, self.model_path, verbose=True)

        def load(self):
                if not os.path.exists(self.model_path):
                        raise ValueError('Can`t find trained word2vec file')
                global word2vec
                model = word2vec.load(self.model_path)
                return model

        def word_embedding(self, vocabs):
                global word2vec
                model = word2vec.load(self.model_path)
                word2vec_numpy = list()
                for word in vocabs:
                        if word in model.vocab:
                                word2vec_numpy.append(model[word].tolist())
                embed_size = len(word2vec_numpy[0])
                col = []
                for i in range(embed_size):
                        count = 0.0
                        for j in range(int(len(word2vec_numpy))):
                                count += word2vec_numpy[j][i]
                                count = round(count, 6)
                        col.append(count)
                zero = []
                for m in range(embed_size):
                        avg = col[m] / (len(word2vec_numpy))
                        avg = round(avg, 6)
                        zero.append(float(avg))
                list_word2vec = []
                oov = 0
                iov = 0
                for word in vocabs:
                        if word not in model.vocab:
                                oov += 1
                                word2vec = zero
                        else:
                                iov += 1
                                word2vec = model[word].tolist()
                        list_word2vec.append(word2vec)
                embed_size = len(list_word2vec[0])
                return list_word2vec, embed_size