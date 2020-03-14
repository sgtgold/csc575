#!/usr/bin/env python3
#!/usr/bin/env python3
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from packages import data
from time import time

def cluster(matrix, distance_measure, linkage_type):

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

    for linkage in ('ward', 'average', 'complete', 'single'):
        cluster = AgglomerativeClustering(linkage=linkage, n_clusters=3)
        t0 = time()
        cluster.fit_predict(matrix)
        plt.title(linkage)
        plt.scatter(matrix.iloc[:,0],matrix.iloc[:,1], c=cluster.labels_, cmap='rainbow')
        print("%s :\t%.2fs" % (linkage, time() - t0))
        plt.show()