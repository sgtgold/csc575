#!/usr/bin/env python3
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial
import time
import pickle
from packages import data
import numpy as np

def kmeans(tfidf_matrix,picklePath,k):
    start_time = time.time()
    print(start_time)
    num_clusters = k
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    pickle.dump(km, open(picklePath, "wb"))
    km = data.readPickle(picklePath)
    clusters = km.labels_.tolist()
    
    dists = euclidean_distances(km.cluster_centers_)
    tri_dists = dists[np.triu_indices(5, 1)]
    max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
    print ("Maximum Distance Between Clusters", max_dist)
    print ("Average  Distance Between Clusters", max_dist)

    # Compute on the upper (or lower) triangular corner of the distance matrix:

    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    #print(clusters)
