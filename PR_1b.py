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
from numpy.linalg import norm
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import matrix_rank
from numpy.linalg import inv
from math import sqrt

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
	
######################################### MAIN STARTS HERE ###########################################################	
def main():	
	#### LOAD FACE DATA
	face_data,face_label = load_face_data('face(1).mat')

	#### PARTITION DATA INTO TRAIN AND TEST SET
	X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')
	
	#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
	original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
	original_test, norm_test = get_original_normalized_feature_vectors(X_test,show='no')
	
	#### CALCULATE EUCLIDIAN DISTANCE BETWEEN VECTORS
	def get_euclidian_distance_array(query_image,image_list):
		euclidian_distance_array = []
		
		for i in range (0,image_list.shape[1]):
			query_image = query_image.astype(float)
			image = image_list[:,i].astype(float)
			
			subtract_elements = np.subtract(query_image,image)
			subtract_elements_squared = np.multiply(subtract_elements,subtract_elements)
			sum = np.sum(subtract_elements_squared)
			euclidian_distance = sum**0.5
			euclidian_distance_array.append(np.round(euclidian_distance,3))
		
		return np.array(euclidian_distance_array)
	
	query_index = 0										# uses first column as the query image, index = 0
	query_image = X_test[:,query_index]					# uses first column as the query image
	image_list = np.delete(X_test,query_index,1)		# removes first column from the image list
	
	query_label = Y_test[query_index]
	image_label_list = np.delete(Y_test,query_index,0)
	
	euclidian_distance_array = get_euclidian_distance_array(query_image,image_list)
	
	#### SORT QUERY FROM LOWEST TO HIGHEST
	idx = euclidian_distance_array.argsort()[::1]
	euclidian_distance_array = euclidian_distance_array[idx]
	query_results = image_label_list[idx]
	
	#### RETRIEVE TOP 'K' RESULTS
	K = 10
	relevant = 0
	top_k_results = []
	for i in range(0,K):
		top_k_results.append(query_results[i])
		if query_label == query_results[i]:
			relevant = relevant + 1
	
	print(relevant)
	print("Precision at rank",K,"=",np.round((relevant/K),3))
	print("Recall at rank",K,"=",np.round((relevant/9),3))

	return 0
	
main()
