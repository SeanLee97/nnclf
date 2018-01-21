# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

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


class CNNBiGRU(object):
    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir.rstrip(os.path.sep)
        self.model, self.start_epoch = Utils.load_previous_model(self.model_dir)

        self.kwargs = kwargs
        '''
        @description: 每次输入的批量数据进行训练，加快训练速度
        @notice: 批处理的大小，适用于大量数据，一般设置为2的n次方且不宜超过128，一般2，4最佳
        '''
        if 'batch_size' not in self.kwargs:
            self.kwargs['batch_size'] = 4
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
        '''
        @description: retrain 是否重复训练，默认False，如果为默认情况则在先前模型基础上继续训练
        '''
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

            self.model = CNNBiGRUNet(self.kwargs)

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
            hidden = self.model.init_hidden()
            output_pred = self.model(input_variable, inputs_len, hidden, training=True)
            loss += self.criterion(output_pred.view(target_variable.size(0), -1), target_variable)
            runing_loss += loss.data[0]
            if self.kwargs['print_log'] is True:
                self.progress.next()
                if epoch % self.kwargs['log_interval'] == 0:
                    
                    _, pred_ids = torch.max(output_pred, dim=1)
                    corrects = (pred_ids.data == target_variable.data)
                    acc = torch.mean(corrects.float())
                    self.progress.log('epoch %d , loss: %.3f, acc %.3f' % (epoch, runing_loss/self.kwargs['log_interval'], acc))
                    
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
                arr.append((self.dataset.all_labels[topi[x]], topv[x]))
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
 CNN-BiGRU网络
'''
class CNNBiGRUNet(nn.Module):
    def __init__(self, kwargs):
        super(CNNBiGRUNet, self).__init__()
        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']
        self.output_size = kwargs['output_size']

        self.n_layers = kwargs.get('n_layers', 1)
        self.batch_size = kwargs.get('batch_size', 1)
        self.embed_size = kwargs.get('embed_size', 100)
        self.kernel_num = kwargs.get('kernel_num', 64)
        self.kernel_sizes = kwargs.get('kernel_sizes', [1, 3, 5])
        self.dropout = kwargs.get('dropout', 0.1)
        
        self.embed = nn.Embedding(self.input_size, self.embed_size)
        
        if 'word_embedding' in kwargs:
            pretrained_weight = torch.from_numpy(kwargs['word_embedding'])
            self.embed.weight.data.copy_(pretrained_weight)
        
        Ci = 1
        Co = self.kernel_num
        Ks = self.kernel_sizes

        # conv2d卷积操作
        self.convs1 = [nn.Conv2d(Ci, Co, (K, self.embed_size), padding=(K//2, 0), stride=(1,1)) for K in Ks]
        self.batch_norm = nn.BatchNorm2d(Co)

        # CNNBiGRU num_directions = 2
        self.bigru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.n_layers, dropout=self.dropout, bidirectional=True, bias=True)
        self.hidden = self.init_hidden()

        # Linear
        L = len(Ks)*Co + self.hidden_size*2  # num_directions = 2
        
        self.h2o = nn.Linear(L, self.output_size)
        self.dropout = nn.Dropout(self.dropout)

    def init_hidden(self):
        # num_directions = 2
        return Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))

    def forward(self, inp, inp_lens, hidden=None, training=False):
        embed = self.embed(inp) # N x batch_size x embed_size
        if training:
            embed = self.dropout(embed)

        inp = embed.view(len(inp), embed.size(1), -1) 

        cnn = inp
        cnn = torch.transpose(cnn, 0, 1) # (batch_size, seq_len, embed_size)
        cnn = cnn.unsqueeze(1) # (batch_size, 1, seq_len, embed_size)
        cnn = [self.batch_norm(F.relu(conv(cnn)).squeeze(3)) for conv in self.convs1] # [(batch_size, seq_len, Co), ...]*len(Ks)
        cnn = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn]    
        cnn = torch.cat(cnn, 1) # 在1轴拼接 (seq_len, Co*len(Ks))

        # BiGRU
        bigru = inp.view(len(inp), inp.size(1), -1) # seq_len x batch_size x embed_size
        bigru = pack_padded_sequence(bigru, inp_lens)
        bigru, hn = self.bigru(bigru, hidden) 
        bigru, output_lens = pad_packed_sequence(bigru)

        # bigru shape -> (seq_len, batch_size, 2*hidden_size)
        bigru = torch.transpose(bigru, 0, 1)    # (batch_size ,seq_len, 2*hidden_size)
        bigru = torch.transpose(bigru, 1, 2)     # (batch_size, 2*hidden_size, seq_len)
        bigru = F.max_pool1d(bigru, bigru.size(2)).squeeze(2)    # (batch_size, 2*hidden_sze)
        
        # CNN BiGRU
        cnn = torch.transpose(cnn, 0, 1) # (Co*len(Ks), batch_size)
        bigru = torch.transpose(bigru, 0, 1) #(2*hidden_size, batch_size)
        cnn_bigru = torch.cat((cnn, bigru), 0) # (Co*len(Ks) + 2*hidden_size, batch_size)
        cnn_bigru = torch.transpose(cnn_bigru, 0, 1) #(batch_size, Co*len(Ks)+2*hidden_size)
        
        output = self.h2o(F.tanh(cnn_bigru))
        if training:
            y = F.log_softmax(output)
        else:
            y = F.softmax(output)
        return y
