# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import jieba
from nnclf.textcnn import TextCNN

def load_data(corpus_dir='./corpus/data/'):
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
clf = TextCNN('./runtime/textcnn', print_log=True, epoch=3, batch_size=1, word2vec=False, word2vec_model='./runtime/word2vec.bin')
clf.train(datas, labels, retrain=True)
texts = ['深圳天气预报', '广州的气温', '我想听你最珍贵这支哥', '放首诗', '放首歌']
labels = clf.predict_proba(texts, k=5)
for idx in range(len(texts)):
	print(texts[idx])
	for label, proba in labels[idx]:
		print(label, ',', proba)
	print()