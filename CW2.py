#!/usr/bin/env python
# coding: utf-8

# In[93]:


###the main
face_data,face_label = load_face_data('face.mat')

#### PARTITION DATA INTO TRAIN AND TEST SET
X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='yes')

#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')

#### EXPRESS AS A DATAFRAME --- P1,P2,P3...P2576,CLASS
df_original = get_df(original_train,Y_train,name='Original',show='yes')
df_norm = get_df(norm_train,Y_train,name='Norm',show='yes')
df_test = get_df(X_test,Y_test,name='Test',show='yes')


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


# In[8]:


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
	



# In[139]:


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
        


# 

# In[72]:


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

    precision = relevant/k * 100
    recall = relevant/9 * 100
    
    return precision, recall


# In[73]:


###the main, continued for KNN retrieval
#calculate euclidean distances between vectors
query_index = 0
#uses first column as the query image, X_text the training data
query_image = X_test[:, query_index]
image_list = np.delete(X_test, query_index, 1)

query_label = Y_test[query_index]
image_label_list = np.delete(Y_test, query_index, 0)

euclid_dist = euclid_dist_array(query_image, image_list)


# In[74]:


#sort query from lowest to highest
idx = euclid_dist.argsort()[::1]
euclid_dist = euclid_dist[idx]
query_results = image_label_list[idx]


# In[78]:


k = 10
print(query_results.shape)

precision, recall = k_NN(query_results, query_label, k)


print("precision at rank", k, "=", np.round(precision, 3), "%")
print("recall at rank", k, "=", np.round(recall, 3), "%")

        


# In[132]:


import matplotlib.pyplot as plt
#converting matrix to a list of array
A_test = []
for i in range(0, X_test.shape[1]):
    A_test.append(X_test[:,i])


### ask the quantisation precision, shall we quantise pixel imtensities based on thier corresponding frequencies
### assume uniform quantisation, for now
### It was found experimentally that test images' pixel intensities are ranged from 0 to approximately, 260
n_bins = 50
intensity_max = 260
bin_width = intensity_max/n_bins
bin_list = [0]

###create list of bin edges for histogram plot
for i in range(0,n_bins):
    next_entry = bin_list[i] + bin_width
    bin_list.append(next_entry)
X_hist = []

for i in range(0,X_test.shape[1]):
    X_hist_test, bins, patches = plt.hist(A_test[i], bins = bin_list)
    title_hist = "histogram for " + str(i+1)
    plt.title(title_hist)
    plt.show()
    X_hist.append(X_hist_test)
    
X_hist = np.array(X_hist)
X_hist = X_hist.T
print("X_hist has shape = ", X_hist.shape)

#describe_matrix(X_hist,'Histogram for test data set')
#describe_list(X_hist[0], 'An array of histogram data')


#print(X_hist_test.shape)


# In[152]:


###the main, continued for KNN retrieval
#calculate euclidean distances between vectors
query_index = 12
#uses first column as the query image, X_text the training data
query_image = X_hist[:, query_index]
image_list = np.delete(X_hist, query_index, 1)

query_label = Y_test[query_index]
image_label_list = np.delete(Y_test, query_index, 0)

euclid_dist = euclid_dist_array(query_image, image_list)

#sort query from lowest to highest
idx = euclid_dist.argsort()[::1]
euclid_dist = euclid_dist[idx]
query_results = image_label_list[idx]

k = 10

#print(query_results.shape)

precision, recall = k_NN(query_results, query_label, k)


print("precision at rank", k, "=", np.round(precision, 3), "%")
print("recall at rank", k, "=", np.round(recall, 3), "%")


# In[ ]:




