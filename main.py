#!/usr/bin/env python3
#TODO: create requirments.txt
#TODO: create readme
from packages import data
from packages import Clustering
import numpy as np


#from packages import clustering
delim = '^~'
sourcePath = './data/raw_data.csv'
destPath = './data/tweets.csv'
tokenPath = './data/tokens.csv'
nmfPath = './data/nmf.pickle'
tfidfPath = './data/tfidf_matrix.pickle'
vectorPath = './data/tfidf_vector.pickle'
kPicklePath = './data/kmeans.pickle'
featurePickelPath = './data/features.pickle'
svdPicklePath = './data/svd.pickle'
num_topics = 10

data.cleanFile(sourcePath,destPath)
data.readFileCreateTFIDF(destPath,tokenPath,tfidfPath,vectorPath,featurePickelPath,delim)
tfidf_matrix = np.matrix(data.readPickle(tfidfPath).toarray())
M,svd = data.ApplySVD(tfidf_matrix,7)
feat_array = data.readPickle(featurePickelPath)
print(feat_array)
#model = data.extractTopics(tfidf_matrix,nmfPath,num_topics)
#data.display_topics(model,feat_array,num_topics)
#kPicklePath = './data/kmeans_svd.pickle'
#Clustering.kmeans(svd,kPicklePath,5)
