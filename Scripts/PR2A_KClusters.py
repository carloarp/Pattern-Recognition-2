import numpy as np
import scipy.io as sc
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import precision_recall_curve
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from metric_learn import LMNN, RCA
from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix
from numpy.linalg import matrix_rank

def describe_matrix(matrix,name):
	rank = matrix_rank(matrix)
	print(name,"has shape",matrix.shape,"and rank",rank,":\n",matrix,"\n")

def getAccuracy(ytrain, labels):
	m = Munkres()									# Hungarian algorithm
	cmat = contingency_matrix(ytrain, labels+1)		
	print(cmat)
	print(cmat.shape)
	mapping = m.compute(cmat.max() - cmat)			# Maps old label to new labels 
	
	print("Hungarian Algorithm Mapping:")
	print(mapping,'\n')
	
	di = {}
	for a, b in mapping: 							# Creates a dictionary according to the mapping
		di.setdefault(b, []).append(a) 
	
	print("Hungarian Algorithm Mapping Dictionary:")
	print(di)
	
	new_labels = []
	for i in labels:
		if i in di:
			new_labels.append(di[i][0])
		else:
			new_labels.append(35)
	new_labels = np.asarray(new_labels).ravel() + 1
	
	print(len(new_labels))
	sys.exit()
	
	accuracy = 0

	for x in range(len(new_labels)):
		if new_labels[x] == ytrain[x]:
			accuracy +=1
	
	return (accuracy/320)

data = sc.loadmat('face(1).mat')

# N: number of images
# D: number of pixels
X = data['X']  # shape: [D x N]
y = data['l']  # shape: [1 x N]

# Number of images
D, N = X.shape

# Fix the random seed
np.random.seed(13)

# We have 52 classes and we want to achieve a split of 320 (classes 1-32) and 200 (classes 33-52)
Xtrain = []
Xtest = []
ytrain = []
ytest = []

for x in range(N):
    if x < 320:
        Xtrain.append(X[:, x])
        ytrain.append(y[:, x])
    else:
        Xtest.append(X[:, x])
        ytest.append(y[:, x])

Xtrain_norm = preprocessing.normalize(Xtrain, norm='l2', axis=1)
Xtest_norm = preprocessing.normalize(Xtest, norm='l2', axis=1)

from sklearn.cluster import KMeans
#from scipy.optimize import linear_sum_assignment

# Number of clusters
kmeans = KMeans(n_clusters=33, n_init=10, init='random')

# Fitting the input data
kmeans = kmeans.fit(Xtrain)
# Getting the cluster label
labels = kmeans.labels_
print(labels,'\n')

# Centroid values
centroids = kmeans.cluster_centers_

acc = getAccuracy(ytrain, labels)
print(acc)