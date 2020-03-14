#!/usr/bin/env python3
#TODO: create readme
from packages import data
from packages import clustering
from packages import hierarchical
import numpy as np
import scipy
import sklearn
import pandas as pd
from tabulate import tabulate

delim = '^~'
sourcePath = './data/raw_data.csv'
tokenPath = './data/tokens.csv'
nmfPath = './data/nmf.pickle'
tfidfPath = './data/tfidf_matrix.pickle'
tfidfPath_small = './data/tfidf_matrix_small.pickle'
vectorPath = './data/tfidf_vector.pickle'
kPicklePath = './data/kmeans.pickle'
featurePickelPath = './data/features.pickle'
svdPicklePath = './data/svd.pickle'
num_topics = 10

#data.cleanFile(sourcePath,destPath,delim)
#data.readFileCreateTFIDF(destPath,tokenPath,tfidfPath,vectorPath,featurePickelPath,delim)
#tfidf_matrix = data.readPickle(tfidfPath)
#M,svd = data.ApplySVD(tfidf_matrix,7)
#feat_array = data.readPickle(featurePickelPath)
#model = data.extractTopics(tfidf_matrix,nmfPath,num_topics)
#data.display_topics(model,feat_array,num_topics)
#kPicklePath = './data/kmeans_svd.pickle'
#Clustering.kmeans(svd,kPicklePath,5)
#tfidf_matrix_small = data.readPickle(tfidfPath_small)
svd_df = pd.DataFrame(pd.read_pickle(svdPicklePath))
print(svd_df.shape)
print(svd_df.head())
hierarchical.singleLink(svd_df.sample(n=int(svd_df.shape[0] * 0.8)))



