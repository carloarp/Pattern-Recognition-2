#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


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


# In[2]:


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

#into dataframe
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
	



# In[7]:


#top k euclidean distance array
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
        


# 

# In[8]:


###the main
face_data,face_label = load_face_data('face.mat')

#### PARTITION DATA INTO TRAIN AND TEST SET
X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')

#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
original_test, norm_test = get_original_normalized_feature_vectors(X_test,show = 'no')


#### EXPRESS AS A DATAFRAME --- P1,P2,P3...P2576,CLASS
df_original = get_df(original_train,Y_train,name='Original',show='no')
df_norm = get_df(norm_train,Y_train,name='Norm',show='no')
df_test = get_df(X_test,Y_test,name='Test',show='no')


# In[9]:


#precision and recall with KNN
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


# In[ ]:





# In[ ]:





# In[11]:


#create list of rank@k
rank_k = []
for i in range(0,199):
    rank_k.append(i+1)
    
#initialise maP
avg_prec = 0
for query_index in range(0,200):
    query_image = X_test[:, query_index]
    image_list = np.delete(X_test, query_index, 1)
    query_label = Y_test[query_index]
    image_label_list = np.delete(Y_test, query_index, 0)

    euclid_dist = euclid_dist_array(query_image, image_list)

    #sort query from lowest to highest
    idx = euclid_dist.argsort()[::1]
    euclid_dist = euclid_dist[idx]
    query_results = image_label_list[idx]

    precision_list = []
    recall_list = []
    #print(query_results.shape)

    ##find accuracy scores for k nearest neighbours
    for k in range(1,200):
        precision, recall = k_NN(query_results, query_label, k)
        precision_list.append(precision)
        recall_list.append(recall)

    eval_metric_table = get_df_score(rank_k, precision_list, recall_list)
    avg_prec = avg_prec + average_precision(eval_metric_table)
    #print(eval_metric_table)
    title = 'total average precision up to image '+ str(query_index+1)
    title = title + '= '
    print(title, avg_prec)
    print('\n')

maP = avg_prec/200
print('mean average precision is ', maP)
        


# In[12]:


import matplotlib.pyplot as plt
import math
#converting matrix to a list of array
A_test = []
for i in range(0, X_test.shape[1]):
    A_test.append(X_test[:,i])


### ask the quantisation precision, shall we quantise pixel imtensities based on thier corresponding frequencies
### assume uniform quantisation, for now
### It was found experimentally that test images' pixel intensities are ranged from 0 to approximately, 260
bin_width = 10
intensity_max = 255
n_bins = math.ceil(intensity_max/bin_width)
bin_list = [0]

###create list of bin edges for histogram plot
for i in range(0,n_bins):
    next_entry = bin_list[i] + bin_width
    bin_list.append(next_entry)
    
X_hist = []

for i in range(0,X_test.shape[1]):
    X_hist_test, bins, patches = plt.hist(A_test[i], bins = bin_list)
    #title_hist = "histogram for " + str(i+1)
    #plt.title(title_hist)
    #plt.show()
    X_hist.append(X_hist_test)
    
X_hist = np.array(X_hist)
X_hist = X_hist.T
print("X_hist has shape = ", X_hist.shape)

#describe_matrix(X_hist,'Histogram for test data set')
#describe_list(X_hist[0], 'An array of histogram data')


#print(X_hist_test.shape)


# In[13]:


###the main, continued for KNN retrieval, for histogram and euclidean distance
#calculate euclidean distances between vectors

#query_index = 0
#uses first column as the query image, X_text the training data
#query_image = X_hist[:, query_index]
#image_list = np.delete(X_hist, query_index, 1)

#query_label = Y_test[query_index]
#image_label_list = np.delete(Y_test, query_index, 0)

#euclid_dist = euclid_dist_array(query_image, image_list)


#initialise maP
avg_prec = 0
for query_index in range(0,200):
    #uses first column as the query image, X_text the training data
    #query_image = X_test[:, query_index]
    #image_list = np.delete(X_test, query_index, 1)

    #query_label = Y_test[query_index]
    #image_label_list = np.delete(Y_test, query_index, 0)

    #euclid_dist = euclid_dist_array(query_image, image_list)
    
    #uses first column as the query image, X_text the training data
    query_image = X_hist[:, query_index]
    image_list = np.delete(X_hist, query_index, 1)

    query_label = Y_test[query_index]
    image_label_list = np.delete(Y_test, query_index, 0)

#euclid_dist = euclid_dist_array(query_image, image_list)

    #sort query from lowest to highest
    idx = euclid_dist.argsort()[::1]
    euclid_dist = euclid_dist[idx]
    query_results = image_label_list[idx]

    precision_list = []
    recall_list = []
    #print(query_results.shape)

    ##find accuracy scores for k nearest neighbours
    for k in range(1,200):
        precision, recall = k_NN(query_results, query_label, k)
        precision_list.append(precision)
        recall_list.append(recall)

    eval_metric_table = get_df_score(rank_k, precision_list, recall_list)
    avg_prec = avg_prec + average_precision(eval_metric_table)
    #print(eval_metric_table)
    title = 'total average precision up to image '+ str(query_index+1)
    title = title + '= '
    print(title, avg_prec)
    print('\n')

maP = avg_prec/200
print('mean average precision is ', maP)
    
#print("precision at rank", k, "=", np.round(precision, 3), "%")
#print("recall at rank", k, "=", np.round(recall, 3), "%")


# In[ ]:





# In[ ]:





# In[14]:


def calculate_mean_image(x_train,train_size):
    sum_of_training_faces = [0]
    for i in range(0,train_size):
        sum_of_training_faces = sum_of_training_faces + x_train[:,i]
    average_training_face = sum_of_training_faces/train_size
    #print("Mean Face has shape",average_training_face.shape,":\n",average_training_face, "\n")
    #print("x_train has shape",x_train.shape,":\n",x_train, "\n")

    return average_training_face


# In[15]:


def calculate_covariance_matrix(x_train, average_training_face, train_size):
    
    A_train = np.subtract(x_train.T, average_training_face)
    A_train = A_train.T
          
    covariance_matrix = np.dot(A_train, A_train.T)/train_size
    
    #print("High-Dimension Covariance Matrix [",np.round(covariance_matrix_time,3), "s ] has shape",high_dimension_covariance_matrix.shape,"and rank",rank_S_hd,":\n",high_dimension_covariance_matrix, "\n")
    #print("Low-Dimension Covariance Matrix [",np.round(low_dimension_covariance_matrix_time,3), "s ] has shape",low_dimension_covariance_matrix.shape,"and rank",rank_S_ld,":\n",low_dimension_covariance_matrix, "\n")

    return covariance_matrix


# In[16]:


###Mahalanobis Distance Learning, compute global covariance matrix of training data
global_mean = calculate_mean_image(original_train, 320)
global_mean_norm = calculate_mean_image(norm_train, 320)

covariance_matrix = calculate_covariance_matrix(original_train, global_mean, 320)
covariance_matrix_norm = calculate_covariance_matrix(norm_train, global_mean_norm, 320)

###since covariance_matrix is singular, therefore find pseudoinverse
similarity = np.linalg.pinv(covariance_matrix)
similarity_norm = np.linalg.pinv(covariance_matrix_norm)


# In[18]:


import scipy as sci
##eigenvalue decomposition
eigenvalues, eigenvectors = eig(similarity) 
#print(eigenvalues)
#print(eigenvectors)
idx_hd = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx_hd]
eigenvectors = eigenvectors[idx_hd]

#choose M eigenvectors
M_list = [16, 32, 64, 128, 256, 2576]

for M in M_list:
    eigenvalue_M = eigenvalues[0:M]
    eigenvalue_matrix = np.zeros((M, M))
    
    for index in range(M):
        eigenvalue_matrix[index,index] = eigenvalue_M[index]
    
    eigenvector_M = eigenvectors[:,0:M]

    G = np.dot(sci.linalg.sqrtm(eigenvalue_matrix), eigenvector_M.T)
    ##take real part of G since imaginary is negligible
    G = np.real(G)

    ###project test data using transformation G for Mahalanobis transformation
    G_xtest = np.dot(G, original_test)
    #G_xtest_norm = np.dot(G, norm_test)
    
    ### K nearest neighbours for projected data and mAP
    
    #initialise maP
    avg_prec = 0
    #kNN for all 200 test images
    for query_index in range(0,200):
        query_image = G_xtest[:, query_index]
        image_list = np.delete(G_xtest, query_index, 1)

        query_label = Y_test[query_index]
        image_label_list = np.delete(Y_test, query_index, 0)

        euclid_dist = euclid_dist_array(query_image, image_list)

        #sort query from lowest to highest
        idx = euclid_dist.argsort()[::1]
        euclid_dist = euclid_dist[idx]
        query_results = image_label_list[idx]

        precision_list = []
        recall_list = []
        #print(query_results.shape)

        ##find accuracy scores for k nearest neighbours
        for k in range(1,200):
            precision, recall = k_NN(query_results, query_label, k)
            precision_list.append(precision)
            recall_list.append(recall)

        eval_metric_table = get_df_score(rank_k, precision_list, recall_list)
        avg_prec = avg_prec + average_precision(eval_metric_table)
        #print(eval_metric_table)
        #title = 'total average precision up to image '+ str(query_index+1)
        #title = title + '= '
        #print(title, avg_prec)
        #print('\n')

    maP = avg_prec/200
    print('mean average precision at M = ', M, 'is ', maP)
 


# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:




