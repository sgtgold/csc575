#!/usr/bin/env python3
#TODO: create requirments.txt
#TODO: create readme
from packages import data
#rom packages import clustering
from packages import clustering
from packages import simCluster

import numpy as np
import time

#from packages import clustering
delim = '^~'
sourcePath = './data/raw_data.csv'
destPath = './data/tweets.csv'
tokenPath = './data/tokens.csv'
nmfPath = './data/nmf.pickle'
tfidfPath = './data/tfidf_matrix.pickle'
tfidfPath = './data/tfidf_matrix_small.pickle'
vectorPath = './data/tfidf_vector.pickle'
kPicklePath = './data/kmeans.pickle'
featurePickelPath = './data/features.pickle'
svdPicklePath = './data/svd.pickle'
#kPicklePath = './data/kmeans_svd.pickle'

simPicklePath = './data/sim.pickle'
num_topics = 10

# data.cleanFile(sourcePath,destPath)
# data.readFileCreateTFIDF(destPath,tokenPath,tfidfPath,vectorPath,featurePickelPath,delim)
start_time = time.time()
tfidf_matrix = np.matrix(data.readPickle(tfidfPath).toarray())
#M,svd = data.ApplySVD(tfidf_matrix,7)
feat_array = data.readPickle(featurePickelPath)
#print(len(feat_array))
#model = data.extractTopics(tfidf_matrix,nmfPath,num_topics)
#data.display_topics(model,feat_array,num_topics)
svd = data.readPickle(svdPicklePath)
clustering.kmeans(svd[:1000],kPicklePath,5)
#Sample of 10000 document SVDs
simCluster.singlepass(svd[:100],0.5,0)
#simCluster.singlepass(svd,0.5,0)

print("--- {:d} sec ---".format(round(time.time() - start_time),0))