# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#
import sys
sys.path.append("..")

import os
import jieba
from nnclf.svm import SVM

def load_data(corpus_dir='../corpus/data/'):
    datas = []
    labels = []
    all_labels = []
    for filename in os.listdir(corpus_dir):
        with open(os.path.join(corpus_dir, filename), 'r') as f:
            for line in f.readlines():
                line = line.strip().replace(' ', '')
                if len(line) == 0:
                    continue
                sentence_list = jieba.lcut(line)

                label = filename.split('.txt')[0]
                datas.append(sentence_list)
                labels.append(label)
    return datas, labels

datas, labels = load_data()
clf = SVM('../runtime')

#clf.cross_validation(datas, labels)

clf.train(datas, labels, C=10)

texts = ['深圳天气预报', '广州的气温', '我想听你最珍贵这支哥', '放首诗', '放首歌']
labels = clf.predict_proba(['讲个故事吧'], k=3)
print(labels)