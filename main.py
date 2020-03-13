#!/usr/bin/env python3
#TODO: create requirments.txt
#TODO: create readme
from packages import data
from packages import clustering
import numpy as np

#from packages import clustering
delim = '^~'
sourcePath = './data/raw_data.csv'
destPath = './data/tweets.csv'
tokenPath = './data/tokens.csv'
picklePath = './data/tfidf_matrix.pickle'
kPicklePath = './data/kmeans.pickle'

svdPicklePath = './data/svd.pickle'
data.cleanFile(sourcePath,destPath)
data.readFileCreateTFIDF(destPath,tokenPath,picklePath,delim)
tfidf_matrix = np.matrix(data.readPickle(picklePath).toarray())
#clustering.kmeans(tfidf_matrix,kPicklePath,5)
svd = data.ApplySVD(tfidf_matrix)
kPicklePath = './data/kmeans_svd.pickle'
clustering.kmeans(svd,kPicklePath,5)

