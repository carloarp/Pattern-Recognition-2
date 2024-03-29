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
from sklearn.decomposition import PCA

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
	for test in range(0,len(label)):
	
		query_label = label[nbrs[1][test][0].astype(int)]
		print("Query label:",query_label, end="\r")
			
		max_precision_list = []
		precision_list = []
		recall_list = []
		K_list = []
				
		for K in range(1,len(label)):
			relevant = 0
				
			for i in range (1,K+1):
				index = nbrs[1][test][i].astype(int)
				query_result = label[index]
					
				if query_label == query_result:
					relevant = relevant + 1
					
			recall = np.round((relevant/9),3)
			precision = np.round((relevant/K),3)
					
			recall_list.append(recall)
			precision_list.append(precision)
			K_list.append(K)
			
		results = pd.DataFrame()
		results['K'] = K_list
		results['Recall'] = recall_list
		results['Precision'] = precision_list 
			
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
		
	print("						",end="\r")
	return np.round(map,3), results
	
######################################### MAIN STARTS HERE ###########################################################
def main():		
	
	#### LOAD FACE DATA
	face_data,face_label = load_face_data('face(1).mat')

	#### PARTITION DATA INTO TRAIN AND TEST SET
	X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')
	
	#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
	original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
	original_test, norm_test = get_original_normalized_feature_vectors(X_test,show='no')
	
	#### COVARIANCE MATRIX CALCULATION
	S_original = np.cov(original_train)			# Calculate covariance matrix
	S_norm = np.cov(norm_train)
	S_inv_original = np.linalg.pinv(S_original)	# Covariance matrix pseudo-inverse
	S_inv_norm = np.linalg.pinv(S_norm)
	
	#### MAHALANOBIS DISTANCE DEFINITION
	def mahalanobis_original(u,v):
		return distance.mahalanobis(u, v, S_inv_original)
	def mahalanobis_norm(u,v):
		return distance.mahalanobis(u, v, S_inv_norm)	
	
	maha_original = NearestNeighbors(n_neighbors=200, metric=mahalanobis_original)
	maha_norm = NearestNeighbors(n_neighbors=200, metric=mahalanobis_norm)
	
	methods = [maha_original,maha_norm]
	method_name = ['Maha Original','Maha Norm'] 
	test_datas = [original_test.T, norm_test.T]
	
	recall_levels = 11
	method_count = 0
	for method in methods:
		
		if method_count == 0:	
			method.fit(test_datas[0])
			method_nbrs = np.asarray(method.kneighbors(test_datas[0]))
			method_map, method_df = calculate_map(method_nbrs, Y_test, recall_levels)
			print(method_name[method_count],"mAP:",method_map)
		
		if method_count == 1:	
			method.fit(test_datas[1])
			method_nbrs = np.asarray(method.kneighbors(test_datas[1]))
			method_map, method_df = calculate_map(method_nbrs, Y_test, recall_levels)
			print(method_name[method_count],"mAP:",method_map)
		
		print("		")
		method_count = method_count + 1
	
	return 0
	

main()