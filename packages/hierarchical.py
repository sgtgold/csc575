#!/usr/bin/env python3
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import time
import pickle
from packages import data

def singleLink(tfidf_matrix):
    start_time = time.time()
    data = tfidf_matrix.values

    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
    cluster.fit_predict(tfidf_matrix)
    plt.scatter(tfidf_matrix.iloc[:,0],tfidf_matrix.iloc[:,1], c=cluster.labels_, cmap='rainbow')
    plt.show()

    #plt.title("Dendograms")
    #dend = shc.dendrogram(shc.linkage(data, method='single'))
    #plt.show()

    print("--- %s minutes ---" % ((time.time() - start_time)/60))