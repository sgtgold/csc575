#!/usr/bin/env python3
#TODO: create readme
from packages import data
from packages import clustering
from packages import hierarchical
import numpy as np
import pandas as pd
from packages import simCluster
delim = '^~'
sourcePath = './data/raw_data.csv'
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

data.cleanFile(sourcePath,tokenPath,delim)
data.readFileCreateTFIDF(tokenPath,tfidfPath,vectorPath,featurePickelPath,delim)
tfidf_matrix = data.readPickle(tfidfPath)
feat_array = data.readPickle(featurePickelPath)
svd = data.LoadSVD(tfidf_matrix,feat_array,svdPicklePath,40)
#data.showSVDPlot(tfidf_matrix)
model = data.extractTopics(tfidf_matrix,nmfPath,num_topics)
data.display_topics(model,feat_array,num_topics)
#kPicklePath = './data/kmeans_svd.pickle'
#clustering.kmeans(svd,kPicklePath,5)
#hierarchical.cluster(pd.DataFrame(pd.read_pickle(svdPicklePath)).T)

#simCluster.singlepass(svd[:10000],0.5,0)
simCluster.singlepass(svd,0.5,0)

