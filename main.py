#!/usr/bin/env python3
#TODO: create requirments.txt
#TODO: create readme
from packages import data    
delim = '^~'
sourcePath = './data/raw_data.csv'
destPath = './data/tweets.csv'
tokenPath = './data/tokens.csv'
picklePath = './data/tfidf_matrix.pickle'
data.cleanFile(sourcePath,destPath)
data.readFileCreateTFIDF(destPath,tokenPath,picklePath,delim)
tfidf_matrix = data.readPickle(picklePath)
data.kmeans(tfidf_matrix,5)
