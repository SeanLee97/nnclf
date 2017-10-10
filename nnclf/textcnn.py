# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
TextCNN
@author Sean
@last modified 2017.09.06 17:14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import time
import jieba
import math
import numpy as np

from datetime import datetime

from nnclf.utils import *

class TextCNN(object):
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

		if self.kwargs['word2vec'] is True:
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

			self.model = TextCNNNet(self.kwargs)

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
			output_pred = self.model(input_variable)
			loss += self.criterion(output_pred.view(target_variable.size(0), -1), target_variable)
			runing_loss += loss.data[0]
			if self.kwargs['print_log'] is True:
				self.progress.next()
				if epoch % self.kwargs['log_interval'] == 0:
					self.progress.log('epoch %d , loss: %f' % (epoch, runing_loss/self.kwargs['log_interval']))
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
			#utils.clip_grad_norm(self.model.parameters(), 5.0)
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
		output = self.model(input_variable)
		topv, topi = output.data.topk(k)
		return topv.squeeze(), topi.squeeze()

	def log(self, string):
		Utils.save_log(string, self.model_dir)

'''
 TextCNN网络
'''
class TextCNNNet(nn.Module):
	def __init__(self, kwargs):
		super(TextCNNNet, self).__init__()
		self.input_size = kwargs['input_size']
		self.hidden_size = kwargs['hidden_size']
		self.output_size = kwargs['output_size']
		if 'kernel_num' in kwargs:
			self.kernel_num = kwargs['kernel_num']
		else:
			self.kernel_num = 128
		if 'kernel_sizes' in kwargs:
			self.kernel_sizes = kwargs['kernel_sizes']
		else:
			self.kernel_sizes = [1, 2, 3, 4]
		if 'embed_size' in kwargs:
			self.embed_size = kwargs['embed_size']
		else:
			self.embed_size = kwargs['hidden_size']		
		if 'dropout' in kwargs:
			self.dropout = kwargs['dropout']
		else:
			self.dropout = 0.1
		if 'wide_conv' in kwargs:
			self.wide_conv = kwargs['wide_conv']
		else:
			self.wide_conv = True
		if 'init_weight' in kwargs:
			self.init_weight = kwargs['init_weight']
		else:
			self.init_weight = False
		if 'init_weight_value' in kwargs:
			self.init_weight_value = kwargs['init_weight_value']
		else:
			self.init_weight_value = 2.0		
		if 'batch_normal' in kwargs:
			self.batch_normal = kwargs['batch_normal']
		else:
			self.batch_normal = False
		if 'batch_normal_momentum' in kwargs:
			self.batch_normal_momentum
		else:
			self.batch_normal_momentum = 0.1
		if 'batch_normal_affine' in kwargs:
			self.batch_normal_affine = kwargs['batch_normal_affine']
		else:
			self.batch_normal_affine = False

		Ci = 1	# input channels, 处理文本,一层通道
		Co = self.kernel_num	# output channel
		Ks = self.kernel_sizes	# list
		
		if 'max_norm' in kwargs:
			self.embed = nn.Embedding(self.input_size, self.embed_size, max_norm=kwargs['max_norm'])
		else:
			self.embed = nn.Embedding(self.input_size, self.embed_size, scale_grad_by_freq=True)
		if 'word_embedding' in kwargs:
			pretrained_weight = torch.from_numpy(kwargs['word_embedding'])
			self.embed.weight.data.copy_(pretrained_weight)
			self.embed.weight.requires_grad = True
		if self.wide_conv is True:
			self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, self.embed_size), stride=(1, 1), padding=(K//2 ,0), dilation=1, bias=True) for K in Ks]
		else:
			self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, self.embed_size), bias=True) for K in Ks]
		if self.init_weight:
			for conv in self.convs1:
				init.xavier_normal(conv.weight.data, gain=np.sqrt(self.init_weight_value))
				fanin, fanout = self.cal_fanin_fanout(conv.weight.data)
				std = np.sqrt(self.init_weight_value) * np.sqrt(2.0 / (fanin+fanout))
				init.uniform(conv.bias, 0, 0)

		self.dropout = nn.Dropout(self.dropout)
		in_fea = len(Ks) * Co

		self.f1 = nn.Linear(in_fea, in_fea//2, bias=True)
		self.f2 = nn.Linear(in_fea//2, self.output_size, bias=True)
		self.h2o = nn.Linear(in_fea, self.output_size)
		self.softmax = nn.LogSoftmax()

		if self.batch_normal:
			self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=self.batch_normal_momentum, affine=self.batch_normal_affine)
			self.f1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=self.batch_normal_momentum, affine=self.batch_normal_affine)
			self.f2_bn = nn.BatchNorm1d(num_features=self.output_size, momentum=self.batch_normal_momentum, affine=self.batch_normal_affine)

	def cal_fanin_fanout(self, tensor):
		dimensions = tensor.ndimension()
		if dimensions < 2:
			raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

		if dimensions == 2:  # Linear
			fan_in = tensor.size(1)
			fan_out = tensor.size(0)
		else:
			num_input_fmaps = tensor.size(1)
			num_output_fmaps = tensor.size(0)
			receptive_field_size = 1
			if tensor.dim() > 2:
				receptive_field_size = tensor[0][0].numel()
			fan_in = num_input_fmaps * receptive_field_size
			fan_out = num_output_fmaps * receptive_field_size
		return fan_in, fan_out

	def forward(self, inp):
		inp = self.embed(inp) # N x W x embed_size -> seq_len x batch_size x embed_size
		inp = torch.transpose(inp, 0, 1) # batch_size x seq_len
		inp = inp.unsqueeze(1) # N x Ci x W x embed_size  -> Ci = 1 --> batch_size x 1 x seq_len x embed_size

		if self.batch_normal is True:
			inp = [self.convs1_bn(F.tanh(conv(inp))).squeeze(3) for conv in self.convs1]	# [(batch_size, seq_len, Co), ...]*len(Ks)
			inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]	# [(N, Co), ...]*len(Ks)
		else:
			inp = [self.dropout(conv(inp).squeeze(3)) for conv in self.convs1]	# [(batch_size, seq_len, Co), ...]*len(Ks)
			inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]	# [(batch_size, seq_len), ...]*len(Ks)

		inp = torch.cat(inp, 1)
		inp = self.dropout(inp)	# (batch_size, len(Ks)*Co)
		if self.batch_normal is True:
			inp = self.f1_bn(self.f1(inp))
			y = self.f2_bn(self.f2(F.tanh(inp)))
		else:
			y = self.h2o(inp)
		y = self.softmax(y)
		return y
