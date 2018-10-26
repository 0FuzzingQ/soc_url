import os
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
#import urlparse
import urllib
from urllib.parse import urlparse
import math
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.utils import shuffle
import gensim
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import LSTM 
from gensim.models import Word2Vec


#warnings.filterwarnings("ignore")

def getlabel(x):
	if x == 0:
		return 0
	elif x == 1:
		return 1


def getw2v(url_list,label_list):
	stop = []
	w2v_list = []
	for i in range(0,url_list.size):
		tmp = []
		name = url_list[i]
		for j in range(0,len(name)):
			tmp.append(name[j])
		w2v_list.append(tmp)

	model = Word2Vec(w2v_list,min_count = 5)
	model.wv.save_word2vec_format('word2vec.txt',binary=False)
	label_vect = []
	wv_vect = []
	for i in range(0,url_list.size):
		name = url_list[i]
		tmp = []
		vect = []
		for j in range(0,len(name)):
			if name[j] in stop:
				continue
			tmp.append(model[name[j]])
			if j >= 49:
				break
		if len(tmp) < 50:
			for k in range(0,50-len(tmp)):
				tmp.append([0]*100)
		vect = np.vstack((x for x in tmp))
		wv_vect.append(vect)
		label_vect.append(label_list[i])
	wv_vect = np.array(wv_vect)
	label_vect = np.array(label_vect)
	return wv_vect,label_vect


if __name__ == '__main__':

	normal_data = pd.read_csv('normal.csv')
	abnormal_data = pd.read_csv("risk.csv")
	normal_data['label'] = normal_data['url'].map(lambda x:getlabel(0)).astype(int)
	abnormal_data['label'] = abnormal_data['url'].map(lambda x:getlabel(1)).astype(int)
	abnormal_data = abnormal_data.drop(['id','risk_type','request_time','http_status','http_user_agent','host','cookie_uid','source_ip','destination_ip','last_update_time'],axis = 1)
	#print normal_data.info()
	#print abnormal_data.info()
	train_data = pd.concat([normal_data,abnormal_data],axis = 0)
	train_data = shuffle(train_data)
	w2v_word_list,label_list = getw2v(train_data['url'].values,train_data['label'].values)

	x_train = w2v_word_list[0:8000]
	y_train = label_list[0:8000]
	x_test = w2v_word_list[8000:]
	y_test = label_list[8000:]

	model = Sequential()
	model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
	model.add(Dense(1,activation='sigmoid'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	print('training now......')
	model.fit(x_train,y_train,nb_epoch=50,batch_size=32)
	print('evalution now......')
	score,acc = model.evaluate(x_test,y_test)
	print(score,acc)
