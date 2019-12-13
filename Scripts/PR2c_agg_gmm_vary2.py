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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
import pdb
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

import warnings 
warnings.filterwarnings('ignore')

np.random.seed(13)

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

    if relevant > 0:
      accuracy = 1
    else:
      accuracy = 0

    precision = relevant/k 
    recall = relevant/9 
    
    return precision, recall, accuracy

def hungarian_algo(train_labels,Y_train, N_clusters):
	#build profit matrix for hungarian computation
	ones = np.ones_like(train_labels)
	train_hungarian = train_labels + ones
	j = 0
	hungarian_matrix = np.ones((N_clusters, N_clusters), dtype = int)
	for i in train_hungarian:
		entry_j = Y_train[j]
		hungarian_matrix[i-1, entry_j-1] += 2
		j += 1
	#need to minimise cost of entry, find cost matrix from profit matrix
	cost_matrix = np.zeros((N_clusters, N_clusters), dtype = int)
	for row in range(N_clusters):
		row_sum = np.sum(hungarian_matrix[row,:])
		for col in range(N_clusters):
			cost_matrix[row, col] = row_sum - hungarian_matrix[row, col]

	  
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	optimised_train_labels = np.zeros_like(train_labels)
	index = 0
	for i in train_labels:
		optimised_train_labels[index] = col_ind[i] + 1
		index += 1
	optimised_train_labels = np.array(optimised_train_labels)

	return optimised_train_labels

def calculate_centroid(original_train, optimised_train_labels, n_cluster):
	#initialise centroid array
	centroid_array =[]
	for j in range(n_cluster):
		find_index = [i for i, value in enumerate(optimised_train_labels) if value == j+1] 
		centroid_sum = [0]
		#print(find_index)
		for M in find_index:
			centroid_sum += original_train[:,M]
		mean_centroid  = centroid_sum/len(find_index)
		centroid_array.append(mean_centroid)
		find_index = []
	  
	return np.array(centroid_array).T

def average_precision(eval_table):
    precision_at_recall = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            x_p = eval_table[eval_table['Recall'] >= recall_level]['Precision']
            interpolated_p = max(x_p)
        except:
            interpolated_p = 0.0
        precision_at_recall.append(interpolated_p)
    avg_prec = np.mean(precision_at_recall)
    
    return avg_prec
	
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
	
def main():	

	print(" ")
	face_data,face_label = load_face_data('face(1).mat')

	#### PARTITION DATA INTO TRAIN AND TEST SET
	X_train,X_test,Y_train,Y_test = partition_data(face_data,face_label,show='no')

	#### OBTAIN ORIGINAL AND NORMALIZED FEATURE VECTORS 
	original_train, norm_train = get_original_normalized_feature_vectors(X_train,show='no')
	original_test, norm_test = get_original_normalized_feature_vectors(X_test,show = 'no')
	
	#### DISTANCE DEFINITIONS
	L1_NN = NearestNeighbors(n_neighbors=200, metric='minkowski', p=1)	#manhattan l1
	L2_NN = NearestNeighbors(n_neighbors=200, metric='minkowski', p=2)	#euclidean l2
	Linf_NN = NearestNeighbors(n_neighbors=200, metric='chebyshev')		#chesboard/chebyshev linf
	cosine_NN = NearestNeighbors(n_neighbors=200, metric='cosine')		#cosine
	corr_NN = NearestNeighbors(n_neighbors=200, metric='correlation')	#correlation
		
	#### VARY NUMBER OF CLUSTERS/GAUSSIANS
	
	N_clusters_list = []
	
	L1_acc1_list	= []	# 0 
	L1_acc10_list	= []
	L1_mAP_list 	= []
		
	L2_acc1_list	= []	# 1
	L2_acc10_list	= []
	L2_mAP_list 	= []
		
	Linf_acc1_list	= []	# 2
	Linf_acc10_list	= []
	Linf_mAP_list 	= []
		
	cos_acc1_list	= []	# 3
	cos_acc10_list	= []
	cos_mAP_list 	= []
		
	corr_acc1_list	= []	# 4
	corr_acc10_list	= []
	corr_mAP_list 	= []
	
	original_train = norm_train
	original_test = norm_test
		
	for N_clusters in range(32,45,1):
	#for N_clusters in range(32,35):
		
		## Initialize data to collect
		mAP_list = []
		rank1_acc_list = []
		rank10_acc_list = []
		
		#cluster = AgglomerativeClustering(n_clusters= x)
		#cluster.fit_predict(original_train.T)
		N_gaussians = N_clusters
		
		#### AGGLOMERATIVE CLUSTERING
		clustering = AgglomerativeClustering(n_clusters = N_clusters, affinity = 'euclidean', linkage = 'ward', distance_threshold = None)
		clustering_train = clustering.fit(original_train.T)
		train_labels = clustering_train.labels_
		train_labels = np.array(train_labels)
		optimised_train_labels = hungarian_algo(train_labels,Y_train,N_clusters)
		
		def create_centroid(optimised_train_labels, N_clusters):
			centroid_array = np.zeros((2576, N_clusters))

			for i in range(len(optimised_train_labels)):
				index = optimised_train_labels[i] - 1
				centroid_array[:, index] += original_train[:, i]

			count_per_centroid = np.zeros(N_clusters)

			for i in range(len(optimised_train_labels)):
				index = optimised_train_labels[i] - 1
				count_per_centroid[index] += 1

			for j in range(centroid_array.shape[1]):
				centroid_array[:,j] = centroid_array[:,j]/count_per_centroid[j]
			
			rank_k = []
			for i in range(0, N_clusters):
				rank_k.append(i+1)
			centroid_label_list = []
			for j in range(0, N_clusters):
				centroid_label_list.append(j+1)
				
			return centroid_array, centroid_label_list
		
		centroid_array, centroid_label_list = create_centroid(optimised_train_labels,N_clusters)
		#describe_matrix(centroid_array,"Centroid Array")
		#describe_list(centroid_label_list,"Centroid Label List")
		
		def fisher_vector(xx, gmm):
			xx = np.atleast_2d(xx)
			N = xx.shape[0]

			# Compute posterior probabilities.
			Q = gmm.predict_proba(xx)  # NxK

			# Compute the sufficient statistics of descriptors.
			Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
			Q_xx = np.dot(Q.T, xx) / N
			Q_xx_2 = np.dot(Q.T, xx ** 2) / N

			# Compute derivatives with respect to mixing weights, means and variances.
			d_pi = Q_sum.squeeze() - gmm.weights_
			d_mu = Q_xx - Q_sum * gmm.means_
			d_sigma = (
				- Q_xx_2
				- Q_sum * gmm.means_ ** 2
				+ Q_sum * gmm.covariances_
				+ 2 * Q_xx * gmm.means_)

			# Merge derivatives into a vector.
			return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
		
		var = []
		for i in range(N_clusters):
			var.append(np.diag(1/(np.cov(original_train.T[clustering_train.labels_ == i].T)+1e-20)))
		var = np.asarray(var)

		priors = []
		for i in range(N_clusters):
			priors.append(len(original_train.T[clustering_train.labels_ == i])/320)
		priors = np.asarray(priors)

		gmm = GaussianMixture(n_components=N_gaussians, covariance_type='diag', means_init=centroid_array.T, precisions_init=var, tol=0.001, reg_covar=1e-3, max_iter=500, n_init=5)
		gmm.fit(original_train.T)
		
		gamma = gmm.predict_proba(original_test.T)
		means = gmm.means_ 
		sigmas = gmm.covariances_
		pis = gmm.weights_
		
		fvtest = []
		for i in range(len(original_test.T)):
			fv = []
			for j in range(N_clusters):
				v = (1/np.sqrt(pis[j]))*gamma[i,j]*((original_test.T[i,:] - means[j, :])/np.sqrt(sigmas[j]))
				u = (1/np.sqrt(2*pis[j]))*gamma[i,j]*((((original_test.T[i,:] - means[j, :])/np.sqrt(sigmas[j])) - 1)**2)
				fv.append(u)
				fv.append(v)
			fvtest.append(fv)

		fvtest = np.asarray(fvtest).reshape(200, 2*N_clusters*2576)
		
		#describe_matrix(fvtest,"Fisher Vectors")
		
		#### BASELINE APPROACH
		methods = [L1_NN, L2_NN, Linf_NN, cosine_NN, corr_NN]
		#methods = [L1_NN]
		method_name = ['L2','L1','Linf','cosine','corr']
		#method_name = ['L1']		
		test_data = fvtest
		
		N_clusters_list.append(N_clusters)
		
		acc1_list		= [L1_acc1_list,  L2_acc1_list,  Linf_acc1_list,  cos_acc1_list,  corr_acc1_list]
		acc10_list		= [L1_acc10_list, L2_acc10_list, Linf_acc10_list, cos_acc10_list, corr_acc10_list]
		mAP_list		= [L1_mAP_list,   L2_mAP_list,   Linf_mAP_list,   cos_mAP_list,   corr_mAP_list]
	
		
		recall_levels = 11
		method_count = 0
		for method in methods:

			method.fit(test_data)
			method_nbrs = np.asarray(method.kneighbors(test_data))
			method_map, method_df, acc1, acc10 = calculate_map(method_nbrs, Y_test, recall_levels)
			print("#Clusters:",N_clusters,",",method_name[method_count],"mAP:",method_map,",Acc@1:",acc1,",Acc@10:",acc10)
			
			#print(method_count)
			
			acc1_list[method_count].append(acc1)
			acc10_list[method_count].append(acc10)
			mAP_list[method_count].append(method_map)
			
			method_count = method_count + 1
		
		print("														")
		
	plt.figure(figsize=(10,10))
	
	x1 = N_clusters_list

	plt.plot(x1, L1_acc1_list,		color = 'red',	 	label = 'L1 Acc@rank1'		, marker = 'o')
	plt.plot(x1, L2_acc1_list,		color = 'blue', 	label = 'L2 Acc@rank1'		, marker = 'o')
	plt.plot(x1, Linf_acc1_list,	color = 'green',	label = 'Linf Acc@rank1'	, marker = 'o')
	plt.plot(x1, cos_acc1_list,		color = 'cyan',		label = 'cos Acc@rank1'		, marker = 'o')
	plt.plot(x1, corr_acc1_list,	color = 'orange',	label = 'corr Acc@rank1'	, marker = 'o')
	
	plt.plot(x1, L1_acc10_list, 	color = 'red',	 	label = 'L1 Acc@rank10'		, marker = 'x')
	plt.plot(x1, L2_acc10_list, 	color = 'blue',		label = 'L2 Acc@rank10'		, marker = 'x')
	plt.plot(x1, Linf_acc10_list,	color = 'green', 	label = 'Linf Acc@rank10'	, marker = 'x')
	plt.plot(x1, cos_acc10_list,	color = 'cyan', 	label = 'cos Acc@rank10'	, marker = 'x')
	plt.plot(x1, corr_acc10_list,	color = 'orange', 	label = 'corr Acc@rank10'	, marker = 'x')
	
	plt.plot(x1, L1_mAP_list,		color = 'red',	 	label = 'L1 mAP'			, marker = 'v')
	plt.plot(x1, L2_mAP_list,		color = 'blue',		label = 'L2 mAP'			, marker = 'v')
	plt.plot(x1, Linf_mAP_list,		color = 'green', 	label = 'Linf mAP'			, marker = 'v')
	plt.plot(x1, cos_mAP_list,		color = 'cyan', 	label = 'cos mAP'			, marker = 'v')
	plt.plot(x1, corr_mAP_list,		color = 'orange', 	label = 'corr mAP'			, marker = 'v')
		
	plt.grid(color = 'black', linestyle = '-', linewidth = 0.1)			# parameters for plot grid
	title_name = 'Norm GMM Performance'
	plt.title(title_name).set_position([0.5,1.05])
	plt.xlabel('# Clusters/Gaussians')
	plt.ylabel('mAP, Accuracy')			
	plt.legend(loc = 'best')

	#plt.savefig(title_name)
	plt.show()
	plt.close()

	sys.exit()
	
main()