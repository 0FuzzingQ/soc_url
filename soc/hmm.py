import os
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import urlparse
import urllib
import math
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.utils import shuffle
from hmmlearn import hmm 

warnings.filterwarnings("ignore")


def getarg(url,tmp_list,tmp_len):
	parsed_tuple = urlparse.urlparse(url)
	url_query = urlparse.parse_qs(parsed_tuple.query,True)
	#print url_query
	#exit()
	for j in url_query.keys():
		if url_query[j][0] and url_query[j][0] != '' and j and j != '':
			vers = []
			strin = len(url_query[j][0])
			for i in range(0,strin):
				c = url_query[j][0][i].lower()
				if ord(c) >= ord('a') and ord(c) <= ord('z'):
					vers.append([ord('A')])
				elif ord(c) >= ord('0') and ord(c) <= ord('9'):
					vers.append([ord('N')])
				else:
					vers.append([ord('C')])
			tmp_list.append(np.array(vers))
			tmp_len.append(strin)
	#print tmp_list,tmp_len
	#exit()


if __name__ == '__main__':
	normal_data = pd.read_csv('normal.csv')
	abnormal_data = pd.read_csv("risk.csv")
	abnormal_data = abnormal_data.drop(['id','risk_type','request_time','http_status','http_user_agent','host','cookie_uid','source_ip','destination_ip','last_update_time'],axis = 1)
	#print normal_data.info()
	#print abnormal_data.info()
	train_data = pd.concat([normal_data,abnormal_data],axis = 0)
	train_data = shuffle(train_data)
	tmp_len = []
	tmp_list = []
	hmm_list =[]
	len_list = []
	#print train_data.url.values[0]
	#exit()
	for i in range(0,len(train_data.url.values)):
		getarg(train_data.url.values[i],tmp_list,tmp_len)

	hmm_list = np.vstack(x for x in tmp_list)
	len_list = np.vstack(x for x in tmp_len)

	model = hmm.GaussianHMM(n_components = 5,n_iter = 1000,tol = 0.01,covariance_type = 'full')
	model.fit(hmm_list,tmp_len)
	print model.transmat_


