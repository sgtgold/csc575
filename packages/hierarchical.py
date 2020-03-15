#!/usr/bin/env python3
#!/usr/bin/env python3
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from packages import data
from time import time

"""
    Agglomerative Clustering

    This performs 4 various ways of hierarchical cluster
    1. ward
    2. average
    3. complete
    4. single

    After performing fit_predit this will display how long the algorithm took.
    This will also show a scatter plot per each algorithm.

"""

def cluster(matrix, number_clusters):

    for linkage in ('ward', 'average', 'complete', 'single'):
        print('Hierarchical clustering - ', linkage)
        cluster = AgglomerativeClustering(linkage=linkage, n_clusters=number_clusters)
        t0 = time()
        cluster.fit_predict(matrix)
        plt.title(linkage)
        plt.scatter(matrix.iloc[:,0],matrix.iloc[:,1], c=cluster.labels_, cmap='rainbow')
        print("End hierarchical clustering for %s :\t%.2fs" % (linkage, time() - t0))

        clusters = {}
        n = 0
        for item in cluster.labels_:
            if item in clusters:
                clusters[item].append(n)
            else:
                clusters[item] = [n]
            n +=1

        for item in clusters:
            items_in_cluster = len(clusters[item])
            cluster_percentage = (items_in_cluster/matrix.shape[0]) * 100.00
            print("Cluster %s \t size: %s \t percentage: %.2f%%" % (item, items_in_cluster, cluster_percentage))

        plt.show()
