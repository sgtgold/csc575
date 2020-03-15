#!/usr/bin/env python3
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial
from packages import data
import matplotlib.pyplot as plt

import time
import pickle
import numpy as np

def kmeans(tfidf_matrix,picklePath,k):
    start_time = time.time()
    print ("\n---Starting K-Means clustering---", time.asctime( time.localtime(start_time)))
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
    
    print("\nNumber of Clusters:",k)
    print ("Maximum Distance Between Clusters", max_dist)
    print ("Average  Distance Between Clusters", avg_dist)
    
    # print('----Clusters---\n')
    # print(clusters)
    plt.plot(dists)
    plt.title('Distances')
    plt.show()
    # plt.scatter(clusters,clusters)
    # plt.show()
    print("\n--- Complete ---", time.asctime( time.localtime(time.time())))
    print("\n\n")