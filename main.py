#!/usr/bin/env python3
#TODO: create readme
from packages import data
from packages import clustering
from packages import hierarchical
import numpy as np
import pandas as pd

delim = '^~'
sourcePath = './data/raw_data.csv'
tokenPath = './data/tokens.csv'
nmfPath = './data/nmf.pickle'
tfidfPath = './data/tfidf_matrix.pickle'
vectorPath = './data/tfidf_vector.pickle'
kPicklePath = './data/kmeans.pickle'
featurePickelPath = './data/features.pickle'
svdPicklePath = './data/svd.pickle'
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
#hierarchical.cluster(pd.DataFrame(pd.read_pickle(svdPicklePath)).sample(n=25000))

