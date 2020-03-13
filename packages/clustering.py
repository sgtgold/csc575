#!/usr/bin/env python3
from sklearn.cluster import KMeans
import time
import pickle
from packages import data

def kmeans(tfidf_matrix,picklePath,k):
    start_time = time.time()
    num_clusters = k
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    pickle.dump(km, open(picklePath, "wb"))
    km = data.readPickle(picklePath)
    clusters = km.labels_.tolist()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print(clusters)