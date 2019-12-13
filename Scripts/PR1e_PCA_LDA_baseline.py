### PCA-LDA
### Varies Mlda while keeping Mpca fixed
### Plots recognition rate

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
	cosine_NN = NearestNeighbors(n_neighbors=200, metric='cosine')		#cosine
	corr_NN = NearestNeighbors(n_neighbors=200, metric='correlation')	#correlation
	
	#### PCA DIMENSION REDUCTION
	methods = [L2_NN, L1_NN, Linf_NN, cosine_NN, corr_NN]
	method_name = ['L2','L1','Linf_NN','cosine_NN','corr_NN'] 
	test_datas = [original_test.T, norm_test.T]
	train_datas = [original_train.T, norm_train.T]
	test_name = ['Original', 'Norm']
	M_pca_list = [16,32,64,128,256]
	data_type = [0,1]
	
	recall_levels = 11
	method_count = 0
	M_lda = 10
	lda = LinearDiscriminantAnalysis(n_components=M_lda)
	for method in methods:
		name_count = 0
		#for test_data in test_datas:
		#for type in data_type:
		
		Mpca_list = []
		mAP_pca_list_original = []
		mAP_lda_list_original = []
		
		acc1_pca_list_original = []
		acc1_lda_list_original = []
		
		acc10_pca_list_original = []
		acc10_lda_list_original = []
		
		
		mAP_pca_list_norm = []
		mAP_lda_list_norm = []
		
		acc1_pca_list_norm = []
		acc1_lda_list_norm = []
		
		acc10_pca_list_norm = []
		acc10_lda_list_norm = []
			
		for M_pca in M_pca_list:
		
			for type in data_type:
				
				pca = PCA(n_components=M_pca)
				train_pca = pca.fit_transform(train_datas[type])
				test_pca = pca.transform(test_datas[type])
			
				train_lda = lda.fit_transform(train_pca,Y_train)
				test_lda = lda.transform(test_pca)
				
				#test_pca = pca.fit_transform(test_datas[type])
				#test_lda = lda.fit_transform(test_pca, Y_test)

				method.fit(test_pca)
				method_nbrs_pca = np.asarray(method.kneighbors(test_pca))
				method_map_pca, method_df_pca, acc1_pca, acc10_pca = calculate_map(method_nbrs_pca, Y_test, recall_levels)
				
				method.fit(test_lda)
				method_nbrs_lda = np.asarray(method.kneighbors(test_lda))
				method_map_lda, method_df_lda, acc1_lda, acc10_lda = calculate_map(method_nbrs_lda, Y_test, recall_levels)
				
				print(method_name[method_count],test_name[name_count],", Mpca =",M_pca,"PCA mAP:",method_map_pca,",Acc@1:",acc1_pca,",Acc@10:",acc10_pca)
				print(method_name[method_count],test_name[name_count],", Mpca =",M_pca,"PCA-LDA mAP:",method_map_lda,",Acc@1:",acc1_lda,",Acc@10:",acc10_lda)

				if type == 0:
					mAP_pca_list_original.append(method_map_pca)
					mAP_lda_list_original.append(method_map_lda)
					acc1_pca_list_original.append(acc1_pca)
					acc1_lda_list_original.append(acc1_lda)
					acc10_pca_list_original.append(acc10_pca)
					acc10_lda_list_original.append(acc10_lda)
					
				if type == 1:
					mAP_pca_list_norm.append(method_map_pca)
					mAP_lda_list_norm.append(method_map_lda)
					acc1_pca_list_norm.append(acc1_pca)
					acc1_lda_list_norm.append(acc1_lda)
					acc10_pca_list_norm.append(acc10_pca)
					acc10_lda_list_norm.append(acc10_lda)

			Mpca_list.append(M_pca)
				
			
		x1 = Mpca_list
		y1 = mAP_pca_list_original
		y2 = mAP_lda_list_original
		y3 = acc1_pca_list_original
		y4 = acc1_lda_list_original
		y5 = acc10_pca_list_original
		y6 = acc10_lda_list_original
		
		y7 = mAP_pca_list_norm
		y8 = mAP_lda_list_norm
		y9 = acc1_pca_list_norm
		y10 = acc1_lda_list_norm
		y11 = acc10_pca_list_norm
		y12 = acc10_lda_list_norm
				
		plt.figure(figsize=(10,10))
		plt.plot(x1, y1, color = 'red', label = 'PCA [O] mAP', marker = 'o')
		plt.plot(x1, y7, color = 'red', label = 'PCA [N] mAP', marker = 'x')
		
		plt.plot(x1, y3, color = 'blue', label = 'PCA [O] Acc@rank1', marker = 'o')
		plt.plot(x1, y9, color = 'blue', label = 'PCA [N] Acc@rank1', marker = 'x')
		
		plt.plot(x1, y5, color = 'green', label = 'PCA [O] Acc@rank10', marker = 'o')
		plt.plot(x1, y11, color = 'green', label = 'PCA [N] Acc@rank10', marker = 'x')
		
		plt.plot(x1, y2, color = 'orange', label = 'PCA-LDA [O] mAP', marker = 'o')	
		plt.plot(x1, y8, color = 'orange', label = 'PCA-LDA [N] mAP', marker = 'x')	
		
		plt.plot(x1, y4, color = 'cyan', label = 'PCA-LDA [O] Acc@rank1', marker = 'o')
		plt.plot(x1, y10, color = 'cyan', label = 'PCA-LDA [N] Acc@rank1', marker = 'x')
		
		plt.plot(x1, y6, color = 'magenta', label = 'PCA-LDA [O] Acc@rank10', marker = 'o')
		plt.plot(x1, y12, color = 'magenta', label = 'PCA-LDA [N] Acc@rank10', marker = 'x')
		
		plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)			# parameters for plot grid
		title_name = str(method_name[method_count]+' Baseline PCA and PCA-LDA Performance')
		plt.title(title_name).set_position([0.5,1.05])
		plt.xlabel('Mpca')
		plt.ylabel('mAP, Accuracy')			
		plt.legend(loc = 'best')
		'''
		for i in range (0,len(y1)):
			txt = str(y1[i])+"[O]"+","+str(y3[i])+"[N]"
			plt.annotate(txt, (x1[i], y1[i]))
			
		for i in range (0,len(y2)):
			txt = str(y2[i])+"[O]"+","+str(y4[i])+"[N]"
			plt.annotate(txt, (x1[i], y2[i]))
		'''
		plt.savefig(title_name)
		#plt.show()
		plt.close()

		
		name_count = name_count + 1
			
		print("		")
		method_count = method_count + 1
	
	
	sys.exit()
	
	#### BASELINE APPROACH
	
	methods = [L2_NN, L1_NN, Linf_NN, cosine_NN, corr_NN]
	method_name = ['L2','L1','Linf_NN','cosine_NN','corr_NN'] 
	test_datas = [original_test.T, norm_test.T]
	test_name = ['Original', 'Norm']
	
	recall_levels = 11
	method_count = 0
	for method in methods:
		name_count = 0
		for test_data in test_datas:
				
			method.fit(test_data)
			method_nbrs = np.asarray(method.kneighbors(test_data))
			method_map, method_df, acc1, acc10 = calculate_map(method_nbrs, Y_test, recall_levels)
			print(method_name[method_count],test_name[name_count],"mAP:",method_map,",Acc@1:",acc1,",Acc@10:",acc10)
			name_count = name_count + 1
		print("		")
		method_count = method_count + 1

	return 0
	
main()