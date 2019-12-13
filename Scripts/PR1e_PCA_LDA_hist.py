############################### IMPORT DEPENDENCIES ######################################################

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
import sys
import time
import os
import psutil
import pandas as pd
import math

from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.linalg import eig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import matrix_rank
from numpy.linalg import inv
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from metric_learn import LMNN, RCA
from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix

import warnings 
warnings.filterwarnings('ignore')

										# comment out this code when running on jupyter
dir = os.getcwd()						# gets the working directory of the file this python script is in
os.chdir (dir)							# changes the working directory to the file this python script is in
print("")								# also change plt.savefig to plt.show

############################### FUNCTIONS START HERE ######################################################
def describe_matrix(matrix,name):
	rank = matrix_rank(matrix)
	print(name,"has shape",matrix.shape,"and rank",rank,":\n",matrix,"\n")
	
def describe_list(list,name):
	length = len(list)
	print(name,"has length",len(list),":\n",list,"\n")

def load_face_data(mat_file):
	## Unpacks the .mat file
	contents = loadmat(mat_file)			
	face_data = contents['X']
	face_label = contents['l']	
	return face_data,face_label

def histogram_intersection(h1, h2):
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])
    return sm

def chi(h1, h2):
    sm = 0
    for i in range(len(h1)):
        sm += ((h1[i] - h2[i])**2)/(h1[i] + h2[i])
    return 0.5*sm

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(q != 0, p * np.log(p / q), 0))

def partition_data(face_data,face_labels,show):
	## Dataset = 520 Images (52 classes, 10 images each)
	## Train Data = First 320 Images (class 1-32)
	## Test Data = Remaining 200 Images (class 33-52)

	x_train = face_data[:,0:320]
	x_test = face_data[:,320:520]
	
	y_train = face_labels[:,0:320].squeeze()
	y_test = face_labels[:,320:520].squeeze()
	
	if show == 'yes':
		describe_matrix(x_train,'Train Face Data (x_train)')
		describe_matrix(x_test,'Test Face Data (x_test)')
		describe_list(y_train,'Train Face Label (y_train)')
		describe_list(y_test,'Test Face Label (y_test)')
	
	return x_train,x_test,y_train,y_test

def get_original_normalized_feature_vectors(X_train,show):
	original_train = X_train
	norm_train = normalize(X_train,axis=0,norm='l2')
	if show == 'yes':
		
		describe_matrix(original_train,'Original Train')
		describe_matrix(norm_train,'Norm Train')
	
	return original_train, norm_train

def get_df(X,Y,name,show):
	df_features = X.T
	df = pd.DataFrame()
	for i in range (0,2576):
		column_name = 'P'+str(i+1)
		df[column_name] = df_features[:,i]

	df['Label'] = Y

	if show == 'yes':
		print(name,'Feature Dataframe:')
		print(df.head(11),'\n')
		print(df.tail(11),'\n')
	
	return df

def plot_df(df1,plot_title,x_axis,y_axis,plot,save):						# function for plotting high-dimension and low-dimension eigenvalues
	y1 = df1[df1.columns[1]]
	x1 = df1[df1.columns[0]]
	plt.figure(figsize=(8,8))
	plt.plot(x1, y1, color = 'red')																
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)			# parameters for plot grid
	plt.title(plot_title).set_position([0.5,1.05])
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)			
	plt.legend(loc = 'best')											# creating legend and placing in at the top right
	if save == 'yes':
		plt.savefig(plot_title)
	if plot == 'yes':
		plt.show()
	plt.close()	

def calculate_map(nbrs, label, recall_levels):
	ap = 0
	acc1_list = []
	acc10_list = []
	for test in range(0,len(label)):
	
		query_label = label[nbrs[1][test][0].astype(int)]
		print(test,". Query label:",query_label, end="\r")
			
		max_precision_list = []
		precision_list = []
		recall_list = []
		K_list = []
		relevant_list = []

		for K in range(1,len(label)):
			relevant = 0
				
			for i in range (1,K+1):
				index = nbrs[1][test][i].astype(int)
				query_result = label[index]
					
				if query_label == query_result:
					relevant = relevant + 1
					
			recall = np.round((relevant/9),3)
			precision = np.round((relevant/K),3)
			
			if K == 1:
				if relevant > 0:
					accuracy = 1
				else:
					accuracy = 0
				acc1_list.append(accuracy)
				
			if K == 10:
				if relevant > 0:
					accuracy = 1
				else:
					accuracy = 0
				acc10_list.append(accuracy)
			'''	
			if K == 1:
				acc1_list.append(accuracy)
			if K == 10:
				acc10_list.append(accuracy)
			'''
			relevant_list.append(relevant)
			recall_list.append(recall)
			precision_list.append(precision)
			K_list.append(K)
			
		results = pd.DataFrame()
		results['K'] = K_list
		results['Relevant'] = relevant_list
		results['Recall'] = recall_list
		results['Precision'] = precision_list 
		
		#print(results)
		#sys.exit()
		
		recall_level_list = []
		max_precision_list = []
		for recall_level in np.linspace(0.0,1.0,recall_levels):
			precision_at_rec = []
			precisions = results[results['Recall']>=recall_level]['Precision']
			max_precision = max(precisions)
				
			max_precision_list.append(max_precision)
			recall_level_list.append(np.round(recall_level,1))
			
		avg_precision = np.mean(max_precision_list)
		
		ap = ap + avg_precision
		
	map = ap/len(label)
	acc1_list = np.array(acc1_list)
	acc10_list = np.array(acc10_list)
	acc1 = np.mean(acc1_list)
	acc10 = np.mean(acc10_list)
	
	
	print("						",end="\r")
	return np.round(map,3), results, acc1, acc10
	
######################################### MAIN STARTS HERE ###########################################################
def main():		
	
	#### LOAD FACE DATA
	face_data,face_label = load_face_data('face(1).mat')

	#### PARTITION DATA INTO TRAIN AND TEST SET
	X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')
	
	#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
	original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
	original_test, norm_test = get_original_normalized_feature_vectors(X_test,show='no')
	
	#### DISTANCE DEFINITIONS
	L1_NN = NearestNeighbors(n_neighbors=200, metric='minkowski', p=1)	#manhattan l1
	L2_NN = NearestNeighbors(n_neighbors=200, metric='minkowski', p=2)	#euclidean l2
	Linf_NN = NearestNeighbors(n_neighbors=200, metric='chebyshev')		#chesboard/chebyshev linf

	earthmover = NearestNeighbors(n_neighbors=200, metric=wasserstein_distance) #wassterstein
	intersection = NearestNeighbors(n_neighbors=200, metric=histogram_intersection)  #intersection
	chisquare = NearestNeighbors(n_neighbors=200, metric=chi)
	kldiv = NearestNeighbors(n_neighbors=200, metric=kl)
	js = NearestNeighbors(n_neighbors=200, metric=distance.jensenshannon)
	
	#### HISTOGRAM
	A_test = []
	for i in range(0, X_test.shape[1]):
		A_test.append(X_test[:,i])
		
	A_train = []
	for i in range(0, X_train.shape[1]):
		A_train.append(X_train[:,i])

	bin_width = 10
	intensity_max = 255
	n_bins = math.ceil(intensity_max/bin_width)
	
	bin_list = np.arange(0, 270, 10).tolist()	# Create a bin list from 0-260
												# It was found empirically that test images' pixel intensities ranged from 0 to ~260
												# Assuming uniform quantisation
	
	print("List of bins:",'\n',bin_list,'\n')
	
	X_hist_test = []
	for i in range(0,X_test.shape[1]):
		X_hist, bins, patches = plt.hist(A_test[i], bins = bin_list)
		X_hist_test.append(X_hist)
		plt.close()
	
	X_hist_train = []
	for j in range(0,X_train.shape[1]):
		X_hist, bins, patches = plt.hist(A_train[j], bins = bin_list)
		X_hist_train.append(X_hist)
		plt.close()
	plt.close()
		
	X_hist_test = np.asarray(X_hist_test)
	X_hist_train = np.asarray(X_hist_train)

	methods = [L2_NN, L1_NN, Linf_NN, earthmover, intersection, chisquare, kldiv, js]
	method_name = ['L2','L1','Linf_NN','Earthmover','Intersection','Chi-Square','K-L Divergence','JS'] 
	test_datas = [X_hist_test]
	train_datas = [X_hist_train] 
	test_name = ['Histogram']
	M_pca_list = [16,32,64,128,256]
	M_pca_list = [4,8,12,16,22,26]		# max of 26 as there are 26 bins
	data_type = [0,1]
	
	recall_levels = 11
	M_lda = 10	
	lda = LinearDiscriminantAnalysis(n_components=M_lda)
	
	method_count = 0
	for method in methods:

		#for test_data in test_datas:
		#for type in data_type:
						
		Mpca_list = []
		mAP_pca_list = []
		mAP_lda_list = []
		
		acc1_pca_list = []
		acc1_lda_list = []
			
		acc10_pca_list = []
		acc10_lda_list = []
			
		for M_pca in M_pca_list:
			
			#pca = PCA(n_components=M_pca)
			#lda = LinearDiscriminantAnalysis(n_components=M_lda)
			
			#test_pca = pca.fit_transform(test_data)
			#test_lda = lda.fit_transform(test_pca, Y_test)
				
			pca = PCA(n_components=M_pca)
			#train_pca = pca.fit_transform(train_datas[0])
			#test_pca = pca.transform(test_datas[0])
			
			train_pca = pca.fit_transform(X_hist_train)
			test_pca = pca.transform(X_hist_test)
			
			train_lda = lda.fit_transform(train_pca,Y_train)
			test_lda = lda.transform(test_pca)
				
			method.fit(test_pca)
			method_nbrs_pca = np.asarray(method.kneighbors(test_pca))
			method_map_pca, method_df_pca, acc1_pca, acc10_pca = calculate_map(method_nbrs_pca, Y_test, recall_levels)
				
			method.fit(test_lda)
			method_nbrs_lda = np.asarray(method.kneighbors(test_lda))
			method_map_lda, method_df_lda, acc1_lda, acc10_lda = calculate_map(method_nbrs_lda, Y_test, recall_levels)
				
			#print(method_name[method_count],test_name[name_count],", Mpca =",M_pca,"PCA mAP:",method_map_pca)
			#print(method_name[method_count],test_name[name_count],", Mpca =",M_pca,"PCA-LDA mAP:",method_map_lda)
				
			print(method_name[method_count],", Mpca =",M_pca,"PCA mAP:",method_map_pca,",Acc@1:",acc1_pca,",Acc@10:",acc10_pca)
			print(method_name[method_count],", Mpca =",M_pca,"PCA-LDA mAP:",method_map_lda,",Acc@1:",acc1_lda,",Acc@10:",acc10_lda)
				
			Mpca_list.append(M_pca)
			mAP_pca_list.append(method_map_pca)
			mAP_lda_list.append(method_map_lda)
				
			acc1_pca_list.append(acc1_pca)
			acc1_lda_list.append(acc1_lda)
			acc10_pca_list.append(acc10_pca)
			acc10_lda_list.append(acc10_lda)
			
		x1 = Mpca_list
		y1 = mAP_pca_list
		y2 = mAP_lda_list
		y3 = acc1_pca_list
		y4 = acc1_lda_list
		y5 = acc10_pca_list
		y6 = acc10_lda_list
				
		plt.figure(figsize=(10,10))
		
		plt.plot(x1, y1, color = 'red', label = 'PCA mAP', marker = 'o')
		plt.plot(x1, y2, color = 'red', label = 'PCA-LDA mAP', marker = 'x')		
		
		plt.plot(x1, y3, color = 'blue', label = 'PCA Acc@rank1', marker = 'o')
		plt.plot(x1, y4, color = 'blue', label = 'PCA-LDA Acc@rank1', marker = 'x')
		
		plt.plot(x1, y5, color = 'green', label = 'PCA Acc@rank10', marker = 'o')
		plt.plot(x1, y6, color = 'green', label = 'PCA-LDA Acc@rank10', marker = 'x')
			
		plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)			# parameters for plot grid
		title_name = str(method_name[method_count]+" "+test_name[0]+' PCA and PCA-LDA Performance')
		plt.title(title_name).set_position([0.5,1.05])
		plt.xlabel('Mpca')
		plt.ylabel('mAP, Accuracy')			
		plt.legend(loc = 'best')
			
		'''
		for i, txt in enumerate(y1):
			plt.annotate(txt, (x1[i], y1[i]))
		for i, txt in enumerate(y2):
			plt.annotate(txt, (x1[i], y2[i]))	
		'''
			
		plt.savefig(title_name)
		#plt.show()
		plt.close()
			
		print("		")
		method_count = method_count + 1
	

main()