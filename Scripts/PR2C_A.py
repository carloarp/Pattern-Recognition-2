

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
from sklearn.neighbors import NearestNeighbors

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
        euclid_dist_array.append(euclid_dist)
    
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

    if relevant > 0:
      accuracy = 1
    else:
      accuracy = 0

    precision = relevant/k 
    recall = relevant/9 
    
    return precision, recall, accuracy
###the main
face_data,face_label = load_face_data('face(1).mat')

#### PARTITION DATA INTO TRAIN AND TEST SET
X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')

#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
original_test, norm_test = get_original_normalized_feature_vectors(X_test,show = 'no')


def hungarian_algo(train_labels, Y_train, n):
  import munkres
  from sklearn.metrics.cluster import contingency_matrix
  from munkres import Munkres
  m = Munkres()
  #build profit matrix for hungarian computation
  
  cmat = contingency_matrix(Y_train, train_labels+1)		
  mapping = m.compute(cmat.max() - cmat)

  #need to minimise cost of entry, find cost matrix from profit matrix

  
 
  di = {}
  for a, b in mapping: 							# Creates a dictionary according to the mapping
    di.setdefault(b, []).append(a) 
	
	#print("Hungarian Algorithm Mapping Dictionary:")
	#print(di)
	
  optimised_train_labels = []
  for i in train_labels:
    if i in di:
      optimised_train_labels.append(di[i][0])
		#else:
			#optimised_train_labels.append(35)
  optimised_train_labels = np.asarray(optimised_train_labels).ravel() + 1
  
  return optimised_train_labels

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
			if K == 10:
				if relevant > 0:
					accuracy = 1
				else:
					accuracy = 0
				
			if K == 1:
				acc1_list.append(accuracy)
			if K == 10:
				acc10_list.append(accuracy)
			
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

from sklearn.cluster import AgglomerativeClustering

L1_NN = NearestNeighbors(n_neighbors=200, metric='minkowski', p=1)	#manhattan l1
L2_NN = NearestNeighbors(n_neighbors=200, metric='minkowski', p=2)	#euclidean l2
Linf_NN = NearestNeighbors(n_neighbors=200, metric='chebyshev')		#chesboard/chebyshev linf
cosine_NN = NearestNeighbors(n_neighbors=200, metric='cosine')		#cosine
corr_NN = NearestNeighbors(n_neighbors=200, metric='correlation')	#correlation
#print(original_train.shape)
n_list = [32, 35, 40, 45, 50, 55, 60]
#n_list = [32, 33, 34]

mAP_l2 = []
acc1_l2=[]
acc10_l2 =[]

mAP_cos = []
acc1_cos=[]
acc10_cos =[]
  
mAP_corr = []
acc1_corr =[]
acc10_corr =[]

for n in n_list:
  clustering = AgglomerativeClustering(n_clusters = n, affinity = 'euclidean', linkage = 'ward', distance_threshold = None)
  clustering_train = clustering.fit(original_train.T)
  train_labels = clustering_train.labels_
#n_connected = clustering_train.n_connected_components_
  train_labels = np.array(train_labels)

  optimised_train_labels = hungarian_algo(train_labels, Y_train, n)
#print(optimised_train_labels)

  centroid_array = np.zeros((2576, 32))

  for i in range(len(optimised_train_labels)):
    index = optimised_train_labels[i] - 1
    centroid_array[:, index] += original_train[:, i]

  count_per_centroid = np.zeros(32)

  for i in range(len(optimised_train_labels)):
    index = optimised_train_labels[i] - 1
    count_per_centroid[index] += 1

  for j in range(centroid_array.shape[1]):
    centroid_array[:,j] /=  count_per_centroid[j]

#print(centroid_array.shape)

  rank_k = []
  for i in range(0, 32):
    rank_k.append(i+1)
  centroid_label_list = []
  for j in range(0, 32):
    centroid_label_list.append(j+1)

  #rint(image_label_list)
  #print(Y_train)

  feature_vector = np.zeros((32, 200))
  for i in range(X_test.shape[1]):
    distance_vector = euclid_dist_array(X_test[:,i], centroid_array)
    feature_vector[:, i] = distance_vector.T
  rank_k = []
  for i in range(1, feature_vector.shape[1]):
    rank_k.append(i)
#print(image_label_list.shape)
    
  
  #### BASELINE APPROACH
  methods = [L2_NN, cosine_NN, corr_NN]
		
  method_name = ['L2','cosine_NN','corr_NN'] 
	
  test_datas = [feature_vector.T]
		
  #test_datas = [centroid_array.T]
  #test_name = ['Original', 'Norm']
  #test_name = ['Original','Original']
  
  

  recall_levels = 11
  method_count = 0
  for method in methods:
    for test_data in test_datas:
					
      method.fit(test_data)
      method_nbrs = np.asarray(method.kneighbors(test_data))
      method_map, method_df, acc1, acc10 = calculate_map(method_nbrs, Y_test, recall_levels)
      print('@n_cluster: ', n, ' ', method_name[method_count], "mAP:",method_map,",Acc@1:",acc1,",Acc@10:",acc10)

      if method_count == 0:
        mAP_l2.append(method_map)
        acc1_l2.append(acc1)
        acc10_l2.append(acc10) 
      
      if method_count == 1:
        mAP_cos.append(method_map)
        acc1_cos.append(acc1)
        acc10_cos.append(acc10) 

      if method_count == 2:
        mAP_corr.append(method_map)
        acc1_corr.append(acc1)
        acc10_corr.append(acc10) 
  
  


    print("		")
    method_count = method_count + 1

performance_score = pd.DataFrame()
performance_score['n_clusters'] = n_list
performance_score['acc@rank 1'] = acc1_l2
performance_score['acc@rank 10'] = acc10_l2
performance_score['mAP'] = mAP_l2

performance_score['acc@rank 1'] = acc1_cos
performance_score['acc@rank 10'] = acc10_cos
performance_score['mAP'] = mAP_cos

performance_score['acc@rank 1'] = acc1_corr
performance_score['acc@rank 10'] = acc10_corr
performance_score['mAP'] = mAP_corr
print (performance_score)

x1 = n_list

y1 = acc1_l2
y2 = acc1_cos
y3 = acc1_corr

y4 = acc10_l2
y5 = acc10_cos
y6 = acc10_corr

y7 = mAP_l2
y8 = mAP_cos
y9 = mAP_corr

plt.figure(figsize=(10,10))

plt.plot(x1, y1, color = 'red', label = 'L2 Acc@rank1', marker = 'o')
plt.plot(x1, y2, color = 'red', label = 'Cos Acc@rank1', marker = 'x')	
plt.plot(x1, y3, color = 'red', label = 'Corr Acc@rank1', marker = 'v')

plt.plot(x1, y4, color = 'blue', label = 'L2 Acc@rank10', marker = 'o')
plt.plot(x1, y5, color = 'blue', label = 'Cos Acc@rank10', marker = 'x')	
plt.plot(x1, y6, color = 'blue', label = 'Corr Acc@rank10', marker = 'v')

plt.plot(x1, y7, color = 'green', label = 'L2 mAP', marker = 'o')
plt.plot(x1, y8, color = 'green', label = 'Cos mAP', marker = 'x')	
plt.plot(x1, y9, color = 'green', label = 'Corr mAP', marker = 'v')


#plt.plot(x1, y4, color = 'green', label = 'PCA-LDA Norm[N]', marker = 'x')	
plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)			# parameters for plot grid
title_name = str('Performance scores against number of clusters')
plt.title(title_name).set_position([0.5,1.05])
plt.xlabel('Number of clusters')
plt.ylabel('mAP, Accuracy')			
plt.legend(loc = 'best')
plt.show()
#plt.savefig('L2_distance.jpeg')
plt.close()

