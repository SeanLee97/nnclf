# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import numpy as np
import glob
import os
import random
import pickle
import math
from nnclf.word_vector import WordVector

class Lang(object):
	def __init__(self):
		self.word2index = {'PAD': 0}
		self.n_words = 1

	def index_words(self, sentence_list):
		for word in sentence_list:
			self.index_word(word)

	def index_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.n_words += 1

class Dataset(object):
	def __init__(self, datas, labels, lang, all_labels, batch_size=4, word2vec=False, word2vec_model=None):
		self.datas = datas
		self.labels = labels
		self.lang = lang
		self.all_labels = all_labels
		self.batch_size = batch_size
		self.word_embedding = None
		self.word2vec_model = word2vec_model
		if word2vec:
			self.word_embedding, self.embed_size = self.wordembed()

	def sentence_idx(self, sentence):
		inp = []
		for word in sentence:
			if word in self.lang.word2index:
				inp.append(self.lang.word2index[word])
			else:
				inp.append(0)
		return inp

	def sentence_pad(self, inp, max_len):
		inp = self.sentence_idx(inp)
		return inp + [0 for i in range(max_len-len(inp))]

	def evaluate_pad(self, inp):
		inp = self.sentence_idx(inp)
		inp_lens = [len(inp)]
		inp = [inp]
		return inp, inp_lens

	def wordembed(self):
		if self.word2vec_model == None:
			raise ValueError('请填写word2vec模型存储路径')
		vocab = [k for k,v in self.lang.word2index.items()]
		model = WordVector(self.word2vec_model)
		word_embedding, embed_size = model.word_embedding(vocab)
		word_embedding = np.array(word_embedding)
		return word_embedding, embed_size

	def random_train(self):
		inputs = []
		inputs_len = []
		targets = []
		for i in range(self.batch_size):
			idx = random.randint(0, len(self.labels)-1)
			inputs.append(self.datas[idx])
			targets.append(self.all_labels.index(self.labels[idx]))
		sorteds = sorted(zip(inputs, targets), key=lambda p: len(p[0]), reverse=True)
		inputs, targets = zip(*sorteds)
		inputs_len = [len(x) for x in inputs]
		max_len = max(inputs_len)
		inputs_pad = [self.sentence_pad(item, max_len) for item in inputs]
		input_var = Variable(torch.LongTensor(inputs_pad))
		target_var = Variable(torch.LongTensor(targets))
		return input_var.transpose(0, 1), target_var, inputs_len

import sys, time
class ProgressBar:
	def __init__(self, count = 0, total = 0, width = 50):
		self.count = count
		self.total = total
		self.width = width
	def next(self):
		self.count += 1

	def log(self, s=None):
		sys.stdout.write(' ' * (self.width + 9) + '\r')
		sys.stdout.flush()
		if s != None:
			print(s)
		progress = math.floor(self.width*self.count/self.total)
		sys.stdout.write('%.0f%% : ' % progress)
		sys.stdout.write('{0:3}/{1:3} '.format(self.count, self.total))
		#sys.stdout.write(">" * int(progress)  + "-" * int(self.width - progress) + "\r")
		if progress == self.width:
			sys.stdout.write('\n')
		sys.stdout.flush()

class Utils(object):
	@staticmethod
	def save_log(string, save_dir, filename='log.txt'):
		save_path = os.path.join(save_dir, filename)
		with open(save_path, 'a') as f:
			f.writelines(string.strip('\n') + '\n')

	@staticmethod
	def load_log(string, save_dir, filename='log.txt'):
		save_path = os.path.join(save_dir, filename)
		data = None
		with open(save_path, 'r') as f:
			data = f.readlines()
		return data

	@staticmethod
	def save_model(model, epoch, save_dir, max_keep=5):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		f_list = glob.glob(os.path.join(save_dir, 'model') + '_*.ckpt')
		if len(f_list) >= max_keep + 2:
			epoch_list = [int(i.split('_')[-1].split('.')[0]) for i in f_list]
			to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
			for f in to_delete:
				os.remove(f)
		name = 'model_{}.ckpt'.format(epoch)
		file_path = os.path.join(save_dir, name)
		#torch.save(model.state_dict(), file_path)
		torch.save(model, file_path)

	@staticmethod
	def load_previous_model(save_dir):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		f_list = glob.glob(os.path.join(save_dir, 'model') + '_*.ckpt')
		start_epoch = 1
		model = None
		if len(f_list) >= 1:
			epoch_list = [int(i.split('_')[-1].split('.')[0]) for i in f_list]
			last_checkpoint = f_list[np.argmax(epoch_list)]
			if os.path.exists(last_checkpoint):
				#print('load from {}'.format(last_checkpoint))
				# CNN 不支持参数保存
				#model.load_state_dict(torch.load(last_checkpoint))
				model = torch.load(last_checkpoint)
				start_epoch = np.max(epoch_list)
		return model, start_epoch

	@staticmethod
	def save_param(param, save_dir, filename='param.pkl'):
		with open(os.path.join(save_dir, filename), 'wb') as f:
			pickle.dump(param, f, True)

	@staticmethod
	def load_param(save_dir, filename='param.pkl'):
		param = None
		with open(os.path.join(save_dir, filename), 'rb') as f:
			param = pickle.load(f)
		return param

	@staticmethod
	def remove_models(save_dir):
		f_list = glob.glob(os.path.join(save_dir, 'model') + '_*.ckpt')
		f_list.append(os.path.join(save_dir, 'param.pkl'))
		f_list.append(os.path.join(save_dir, 'log.txt'))
		for filename in f_list:
			try:
				os.remove(filename)
			except:
				pass
