# -*- coding: utf-8 -*-
"""Clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nG1XYamoF6P-66d5POEt43fXYRgWdVkI
"""

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
#import cv2
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

def describe_matrix(matrix,name):
	rank = matrix_rank(matrix)
	print(name,"has shape",matrix.shape,"and rank",rank,":\n",matrix,"\n")
	
def describe_list(list,name):
	length = len(list)
	print(name,"has length",len(list),":\n",list,"\n")

#load face data
def load_face_data(mat_file):
	## Unpacks the .mat file
	contents = loadmat(mat_file)			
	face_data = contents['X']
	face_label = contents['l']	
	return face_data,face_label
	

#partition data into training and test set
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

#normalised feature vectors into unit vectos in L2
def get_original_normalized_feature_vectors(X_train,show):
	original_train = X_train
	norm_train = normalize(X_train,axis=0,norm='l2')
	if show == 'yes':
		
		describe_matrix(original_train,'Original Train')
		describe_matrix(norm_train,'Norm Train')
	
	return original_train, norm_train

def euclid_dist_array(query_image, image_list):
    euclid_dist_array = []
    
    for i in range(0, image_list.shape[1]):
        query_image = query_image.astype(float)
        image = image_list[:,i].astype(float)
        
        subtract_elements = np.subtract(query_image,image)
        subtract_elements_squared = np.dot(subtract_elements, subtract_elements)
        sum = np.sum(subtract_elements_squared)
        euclid_dist = sum**0.5
        euclid_dist_array.append(np.round(euclid_dist, 3))
    
    return np.array(euclid_dist_array)

def average_precision(eval_table):
    precision_at_recall = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            x_p = eval_table[eval_table['recall'] >= recall_level]['precision']
            interpolated_p = max(x_p)
        except:
            interpolated_p = 0.0
        precision_at_recall.append(interpolated_p)
    avg_prec = np.mean(precision_at_recall)
    
    return avg_prec

def get_df_score(rank_k, precision_list, recall_list):
    eval_table = pd.DataFrame()
    eval_table['@rank k'] = rank_k
    eval_table['precision'] = precision_list
    eval_table['recall'] = recall_list
    #AP = average_precision(eval_table)
    
    return eval_table

def k_NN(query_results, query_label, k):
    #retrieve top k results
    #define constants
    relevant = 0
    top_k = []

    #count number of relevant retrievals
    for i in range (0,k):
        top_k.append(query_results[i])
        if query_results[i] == query_label:
            relevant = relevant + 1

    precision = relevant/k 
    recall = relevant/9 
    
    return precision, recall

###the main
face_data,face_label = load_face_data('face.mat')

#### PARTITION DATA INTO TRAIN AND TEST SET
X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')

#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
original_test, norm_test = get_original_normalized_feature_vectors(X_test,show = 'no')

from sklearn.cluster import AgglomerativeClustering

print(original_train.shape)

clustering = AgglomerativeClustering(n_clusters = 32, affinity = 'euclidean', linkage = 'ward', distance_threshold = None)
clustering_train = clustering.fit(original_train.T)
train_labels = clustering_train.labels_
#n_connected = clustering_train.n_connected_components_
train_labels = np.array(train_labels)
print(train_labels)
#print(n_connected)

def hungarian_algo(train_labels):

  #build profit matrix for hungarian computation
  ones = np.ones_like(train_labels)
  #print(ones)
  train_hungarian = train_labels + ones
  #print(train_hungarian)
  j = 0
  hungarian_matrix = np.ones((32, 32), dtype = int)
  for i in train_hungarian:
    entry_j = Y_train[j]
    #print(entry_j)
    hungarian_matrix[i-1, entry_j-1] += 2
    j += 1
  #need to minimise cost of entry, find cost matrix from profit matrix
  cost_matrix = np.zeros((32, 32), dtype = int)
  for row in range(32):
    row_sum = np.sum(hungarian_matrix[row,:])
    for col in range(32):
      cost_matrix[row, col] = row_sum - hungarian_matrix[row, col]

  #print(cost_matrix)
  #print(cost_matrix[1,:])

  #now use library
  from scipy.optimize import linear_sum_assignment
  row_ind, col_ind = linear_sum_assignment(cost_matrix)
  #print(col_ind)
  #print(len(train_labels))

  #new set of labels
  optimised_train_labels = np.zeros_like(train_labels)
  #print(train_labels)
  index = 0
  for i in train_labels:
    optimised_train_labels[index] = col_ind[i] + 1
    index += 1
  optimised_train_labels = np.array(optimised_train_labels)

  #print(optimised_train_labels)
  #print(train_labels)

  return optimised_train_labels

optimised_train_labels = hungarian_algo(train_labels)
print(optimised_train_labels)

#find rank 1 accuracy
k = 1
rank1_prec = np.zeros(320, dtype = int)
rank1_prec_alt = np.zeros(320, dtype = int)
for query_index in range(0,320):
    query_image = original_train[:, query_index]
    image_list = np.delete(original_train, query_index, 1)
    query_label_1 = optimised_train_labels[query_index]
    image_label_list = np.delete(Y_train, query_index, 0)
    query_label_2 = Y_train[query_index]
    image_label_cluster = np.delete(optimised_train_labels, query_index, 0)

    euclid_dist = euclid_dist_array(query_image, image_list)

    #sort query from lowest to highest
    idx = euclid_dist.argsort()[::1]
    euclid_dist = euclid_dist[idx]
    query_results_1 = image_label_list[idx]
    query_results_2 = image_label_cluster[idx]
    
    ##find accuracy scores for nearest neighbour
    precision_1, recall_1 = k_NN(query_results_1, query_label_1, k)
    precision_2, recall_2 = k_NN(query_results_2, query_label_2, k)
    rank1_prec[query_index] = precision_1
    rank1_prec_alt[query_index] = precision_2

rank1_prec = np.array(rank1_prec)
rank1_prec_alt = np.array(rank1_prec_alt)
rank1_prec_avg = np.mean(rank1_prec)
rank1_prec_avg2 = np.mean(rank1_prec_alt)

print('average rank 1 precision is ', (rank1_prec_avg+rank1_prec_avg2)/2, '\n')













