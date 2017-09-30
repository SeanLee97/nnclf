# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
svm 分类器封装
@author Sean
'''

import os
from sklearn.svm import SVC
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
#from nlu.tokenizer import Tokenizer

class SVM(object):
	def __init__(self, model_dir, **kwargs):
		self.model_dir = model_dir.rstrip(os.path.sep)
		self.model_path = os.path.join(self.model_dir, self.__class__.__name__ + '.pkl')
		self.kwargs = kwargs

	def tokenizer(self, inp):
		#return Tokenizer.rule(inp, True)
		import jieba
		return jieba.lcut(inp)

	def train(self, datas, labels, C=1.0, save=True):
		if type(datas[0]) == list:
			datas = [''.join(x) for x in datas]
		vector = TfidfVectorizer(binary=False, tokenizer = self.tokenizer)
		data_vec = vector.fit_transform(datas)
		clf = SVC(kernel='linear', C=C, probability=True)
		clf.fit(data_vec, labels)
		if save:
			import pickle
			with open(self.model_path, 'wb') as f:
				pickle.dump({'clf': clf ,'vector': vector}, f, True)	
		return clf

	def predict(self, datas, clf=None, k=1):
		preds = []
		if clf == None:
			if os.path.exists(self.model_path):
				import pickle
				with open(self.model_path, 'rb') as f:
					save_dict = pickle.load(f)
					clf = save_dict['clf']
					vector = save_dict['vector']

		if clf != None:
			if type(datas) == list and type(datas[0]) == list:
				datas = [''.join(x) for x in datas]
			
			data_vec = [vector.transform([x]) for x in datas]
			for vec in data_vec:
				probas = clf.predict_proba(vec)
				pred = zip(clf.classes_, probas[0].tolist())
				pred = sorted(pred, key=lambda x: x[1], reverse=True)
				if k < len(pred):
					pred = [label for idx, (label, proba) in enumerate(pred) if idx < k]
				preds.append(pred)
		return preds

	def predict_proba(self, datas, clf=None, k=1):
		preds = []
		if clf == None:
			if os.path.exists(self.model_path):
				import pickle
				with open(self.model_path, 'rb') as f:
					save_dict = pickle.load(f)
					clf = save_dict['clf']
					vector = save_dict['vector']
		if clf != None:
			if type(datas) == list and type(datas[0]) == list:
				datas = [''.join(x) for x in datas]
			data_vec = [vector.transform([x]) for x in datas]
			for vec in data_vec:
				probas = clf.predict_proba(vec)
				pred = zip(clf.classes_, probas[0].tolist())
				pred = sorted(pred, key=lambda x: x[1], reverse=True)
				if k < len(pred):
					pred = pred[:k]
				preds.append(dict(pred))
		return preds

	def cross_validation(self, datas, labels, C_list = [0.001, 0.01, 0.1, 1, 10, 100]):
		import numpy as np
		import matplotlib.pyplot as plt
		from sklearn.pipeline import Pipeline
		from sklearn.model_selection import validation_curve

		if type(datas) == list and type(datas[0]) == list:
			datas = [''.join(x) for x in datas]
		
		vector = TfidfVectorizer(binary=False, tokenizer=self.tokenizer)
		data_vec = vector.fit_transform(datas)
		estimator = Pipeline([('clf', SVC(kernel='linear'))])
		train_scores, test_scores = validation_curve(estimator=estimator, X=data_vec, y=labels, param_name='clf__C', param_range=C_list, cv = 10)
		train_mean = np.mean(train_scores, axis=1)
		train_std = np.std(train_scores, axis=1)
		test_mean = np.mean(test_scores, axis=1)
		test_std = np.std(test_scores, axis=1)

		fig = plt.figure(figsize=(15, 6))
		plt.plot(C_list, train_mean,color='blue', marker='o', markersize=5, label='training accuracy')
		plt.fill_between(C_list, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
		plt.plot(C_list, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
		plt.fill_between(C_list, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')
		plt.grid()
		plt.xscale('log')
		plt.legend(loc='lower right')
		plt.xlabel('Parameter C')
		plt.ylabel('Accuracy')
		plt.ylim([0.8, 1.0])
		plt.show() 
