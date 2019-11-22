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

def get_dfs(norm_train,original_train,Y_train,show):
	df_features_norm = norm_train.T
	df_features_original = original_train.T
	df_norm = pd.DataFrame()
	df_original = pd.DataFrame()
	for i in range (0,2576):
		column_name = 'P'+str(i+1)
		df_norm[column_name] = df_features_norm[:,i]
		df_original[column_name] = df_features_original[:,i]
	
	df_norm['Label'] = Y_train
	df_original['Label'] = Y_train
	
	if show == 'yes':
		print('Original Feature Dataframe:')
		print(df_original.head(11),'\n')
		print(df_original.tail(11),'\n')
		print('Norm Feature Dataframe:')
		print(df_norm.head(11),'\n')
		print(df_norm.tail(11),'\n')
	
	return df_original, df_norm

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
	
	#### EXPRESS AS A DATAFRAME --- P1,P2,P3...P2576,CLASS
	df_original = get_df(original_train,Y_train,name='Original',show='no')
	df_norm = get_df(norm_train,Y_train,name='Norm',show='no')
	df_test = get_df(X_test,Y_test,name='Test',show='no')
	
	#### RETRIEVAL KNN
	Xtrain_original = df_original.iloc[:,:-1].values
	Xtrain_norm = df_norm.iloc[:,:-1].values
	Ytrain = df_norm['Label']
	
	Xtest = df_test.iloc[:,:-1].values
	Ytest = df_test['Label']
	
	N=5
	
	KNN_original = KNeighborsClassifier(n_neighbors=N)
	KNN_original.fit(Xtrain_original, Ytrain)
	
	KNN_norm = KNeighborsClassifier(n_neighbors=N)
	KNN_norm.fit(Xtrain_norm, Ytrain)
	
	y_pred_original = KNN_original.predict(Xtest)
	y_pred_norm = KNN_norm.predict(Xtest)

	return 0
	
main()