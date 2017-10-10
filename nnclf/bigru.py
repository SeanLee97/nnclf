# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
BiGRU
@author Sean
@last modified 2017.09.06 17:14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import time
import jieba
import math
import numpy as np

from datetime import datetime

from nnclf.utils import *


class BiGRU(object):
	def __init__(self, model_dir, **kwargs):
		self.model_dir = model_dir.rstrip(os.path.sep)
		self.model, self.start_epoch = Utils.load_previous_model(self.model_dir)

		self.kwargs = kwargs
		'''
		@description: 每次输入的批量数据进行训练，加快训练速度
		@notice: 批处理的大小，适用于大量数据，一般设置为2的n次方且不宜超过128，一般2，4最佳
		'''
		if 'batch_size' not in self.kwargs:
			self.kwargs['batch_size'] = 1
		'''
		@description: 隐藏层大小		
		'''
		if 'hidden_size' not in self.kwargs:
			self.kwargs['hidden_size'] = 64
		'''
		@description: 训练回合数，总的训练次数=回合数x数据集大小		
		'''
		if 'epoch' not in self.kwargs:
			self.kwargs['epoch'] = 3
		'''
		@description: 是否使用word2vec预训练词向量训练
		@notice: 为True时必须传入word2vec_model指定词向量路径		
		'''
		if 'word2vec' not in self.kwargs:
			self.kwargs['word2vec'] = False
		'''
		@description: word2vec模型路径
		'''
		if 'word2vec_model' not in self.kwargs:
			self.kwargs['word2vec_model'] = None
		'''
		@description: 学习速率		
		'''
		if 'lr_rate' not in self.kwargs:
			self.kwargs['lr_rate'] = 0.01
		'''
		@description: 学习速率是否自动进行梯度变化
		'''
		if 'autolr' not in self.kwargs:
			self.kwargs['autolr'] = True
		'''
		@description: 学习速率变化的次数
		'''
		if 'autolr_times' not in self.kwargs:
			self.kwargs['autolr_times'] = 3
		'''
		@description: 是否输出运行记录
		'''
		if 'print_log' not in self.kwargs:
			self.kwargs['print_log'] = False
		'''
		@description: 训练多少次时输出运行记录
		'''
		if 'log_interval' not in self.kwargs:
			self.kwargs['log_interval'] = 100

	def train(self, datas, labels, **kwargs):

		if 'retrain' in kwargs and kwargs['retrain'] is True:
			Utils.remove_models(self.model_dir)

		self.datas = datas
		self.labels = labels

		# 加载之前的模型和步数
		self.model, self.start_epoch = Utils.load_previous_model(self.model_dir)
		lang = Lang()
		all_labels = []
		for idx in range(len(datas)):
			data = datas[idx]
			if type(data) == str:
				data_list = jieba.lcut(data)
			lang.index_words(data)
			if labels[idx] not in all_labels:
				all_labels.append(labels[idx])

		if self.kwargs['word2vec']:
			if self.kwargs['word2vec_model'] == None:
				raise ValueError('请填写word2vec模型存储路径')
			self.dataset = Dataset(datas, labels, lang, all_labels, batch_size=self.kwargs['batch_size'], word2vec=True, word2vec_model=self.kwargs['word2vec_model'])
			if self.model == None:
				self.kwargs['word_embedding'] = self.dataset.word_embedding
				self.kwargs['embed_size'] = self.dataset.embed_size
		else:
			self.dataset = Dataset(datas, labels, lang, all_labels, batch_size=self.kwargs['batch_size'], word2vec=False)

		param = self.kwargs
		param['dataset'] = self.dataset
		Utils.save_param(param, self.model_dir)

		if self.model == None:
			self.kwargs['input_size'] = lang.n_words
			self.kwargs['output_size'] = len(all_labels)
			self.model = BiGRUNet(self.kwargs)

		self.criterion = torch.nn.NLLLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.kwargs['lr_rate'])
		
		self.n_epochs = len(labels) * self.kwargs['epoch']
		self.progress = ProgressBar(count = self.start_epoch, total = self.n_epochs+1)

		self.train_iter()

	def train_iter(self):
		self.log("开始训练时间：%s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
		self.log("总共训练次数：%s" % self.n_epochs)

		self.model.train(True)
		start_time = time.time()
		
		if self.kwargs['autolr'] is True:
			#scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, min_lr=0.000001)
			scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.n_epochs//self.kwargs['autolr_times'], gamma=0.1)

		runing_loss = 0
		for epoch in range(self.start_epoch, self.n_epochs+1):
			loss = 0
			if self.kwargs['autolr'] is True:
				scheduler.step()
				#self.log("learning_rate改变为：%s, 当前训练次数：%d" % (self.optimizer.param_groups[0].get("lr"), epoch))

			self.optimizer.zero_grad()

			input_variable, target_variable, inputs_len = self.dataset.random_train()
			output_pred = self.model(input_variable, inputs_len, self.model.init_hidden())
			loss += self.criterion(output_pred.view(target_variable.size(0), -1), target_variable)
			runing_loss += loss.data[0]
			if self.kwargs['print_log'] is True:
				self.progress.next()
				if epoch % self.kwargs['log_interval'] == 0:
					self.progress.log('epoch %d , loss: %.6f' % (epoch, runing_loss/self.kwargs['log_interval']))
					runing_loss = 0
				else:
					self.progress.log()

			if epoch % (self.n_epochs//10) == 0:
				Utils.save_model(self.model, epoch, self.model_dir)
				log = 'save model :%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				self.log(log)
				if self.kwargs['print_log'] is True:
					print(log)
				
			# update
			loss.backward()
			utils.clip_grad_norm(self.model.parameters(), max_norm=5.0)
			self.optimizer.step()

		# save model
		Utils.save_model(self.model, epoch, self.model_dir)
		time_count = time.time() - start_time
		self.log("结束训练时间： %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
		self.log("总共花费: %.2f小时, 最后loss: %f" % (time_count/3600, runing_loss))

	def predict(self, datas, k=1):
		param = Utils.load_param(self.model_dir)
		self.dataset = param['dataset']

		labels = []
		for data in datas:
			if type(data) == str:
				data = jieba.lcut(data)
			topv, topi = self.evaluate(data, k)
			labels.append([self.dataset.all_labels[x] for x in topi])
		return labels

	def predict_proba(self, datas, k=1):
		label_probas = []
		for data in datas:
			if type(data) == str:
				data = jieba.lcut(data)
			topv, topi = self.evaluate(data, k)
			arr = []
			for x in range(len(topv)):
				arr.append((self.dataset.all_labels[topi[x]], math.exp(topv[x])))
			label_probas.append(arr)
		return label_probas

	def evaluate(self, data, k=1):
		if k > len(self.dataset.all_labels):
			k = len(self.dataset.all_labels)
		inp, inp_lens = self.dataset.evaluate_pad(data)
		input_tensor = torch.LongTensor(inp)
		input_variable = Variable(input_tensor, volatile=True).transpose(0, 1)
		output = self.model(input_variable, inp_lens, None)
		topv, topi = output.data.topk(k)
		return topv.squeeze(), topi.squeeze()

	def log(self, string):
		Utils.save_log(string, self.model_dir)

'''
 BiGRU网络
'''
class BiGRUNet(nn.Module):
	def __init__(self, kwargs):
		super(BiGRUNet, self).__init__()
		self.input_size = kwargs['input_size']
		self.hidden_size = kwargs['hidden_size']
		self.output_size = kwargs['output_size']
		if 'embed_size' in kwargs:
			self.embed_size = kwargs['embed_size']
		else:
			self.embed_size = kwargs['hidden_size']
		if 'batch_size' in kwargs:
			self.batch_size = kwargs['batch_size']
		else:
			self.batch_size = 1
		if 'n_layers' in kwargs:
			self.n_layers = kwargs['n_layers']
		else: 
			self.n_layers = 1
		if 'dropout' in kwargs:
			self.dropout = kwargs['dropout']
		else:
			self.dropout = 0.1
		
		self.embed = nn.Embedding(self.input_size, self.embed_size)
		if 'word_embedding' in kwargs:
			# pretrained embedding
			pretrained_weight = torch.from_numpy(kwargs['word_embedding'])
			self.embed.weight.data.copy_(pretrained_weight)
		self.bigru = nn.GRU(self.embed_size, self.hidden_size, dropout=self.dropout, num_layers=self.n_layers, bidirectional=True)
		# BiGRU num_dicrections = 2
		# full connected
		self.h2o = nn.Linear(self.hidden_size * 2, self.output_size)
		self.hidden = self.init_hidden()
		self.dropout = nn.Dropout(self.dropout)
		self.softmax = nn.LogSoftmax()

	def init_hidden(self):
		return Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))
	
	def forward(self, inp, inp_lens, hidden=None):
		embed = self.embed(inp) # N x batch_size x embed_size
		embed = self.dropout(embed)
		inp = embed.view(len(inp), embed.size(1), -1)
		# pack
		inp = pack_padded_sequence(inp, inp_lens) 
		# reshape
		gru_out, hn = self.bigru(inp, hidden)
		# pad
		gru_out, output_lens = pad_packed_sequence(gru_out)

		# gru_out shape -> (n_layers, batch_size, 2*hidden_size) 
		gru_out = torch.transpose(gru_out, 0, 1)  # (batch_size, n_layers, 2*hidden_size)
		gru_out = torch.transpose(gru_out, 1, 2)  # (batch_size, 2*hidden_size, n_layers)

		gru_out = F.relu(gru_out)
		# max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
		# kernerl_size = (2*n_layers, 2*n_layers)
		gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
		# gru_out shape -> input_size x 2*hidden_size
		gru_out = F.relu(gru_out)
		y = self.softmax(self.h2o(gru_out))
		return y
