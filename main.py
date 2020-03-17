#!/usr/bin/env python3
#TODO: create readme
from packages import data
from packages import clustering
from packages import hierarchical
import numpy as np
import pandas as pd
from packages import simCluster
import sys, getopt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("m",type=str,default='m',help="This selects the memory level for the program the choices are h,m,l for high, medium or low example: py main.py l")
args = parser.parse_args()
low_memory = 10000
medium_memory = 25000
high_memory = 50000
#default
opt = args.m
hierarchical_clustering_memory = medium_memory
if opt == 'l':
    print('Running in Low Memory Mode')
    hierarchical_clustering_memory = low_memory
elif opt == 'm':
    print('Running in Medium Memory Mode')
    hierarchical_clustering_memory = medium_memory
elif opt == 'h':
    print('Running in High Memory Mode')
    hierarchical_clustering_memory = high_memory 
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


    
hierarchical_clustering_number_clusters = 5

data.cleanFile(sourcePath,tokenPath,delim)
data.readFileCreateTFIDF(tokenPath,tfidfPath,vectorPath,featurePickelPath,delim)
tfidf_matrix = data.readPickle(tfidfPath)
feat_array = data.readPickle(featurePickelPath)
svd = data.LoadSVD(tfidf_matrix,feat_array,svdPicklePath,40)
model = data.extractTopics(tfidf_matrix,nmfPath,num_topics)
data.display_topics(model,feat_array,num_topics)

kPicklePath = './data/kmeans_svd.pickle'
clustering.kmeans(svd,kPicklePath,5)
hierarchical.cluster(pd.DataFrame(pd.read_pickle(svdPicklePath)).sample(n=hierarchical_clustering_memory), hierarchical_clustering_number_clusters)
simCluster.singlepass(svd[:10000],0.5,0)
